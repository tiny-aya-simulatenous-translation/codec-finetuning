"""Segmental Signal-to-Noise Ratio (SSNR) measurement.

Computes SSNR between original and reconstructed audio by segmenting
waveforms into overlapping frames and averaging per-segment SNR values,
excluding silent segments.

Performance
-----------
**Parallel per-utterance computation** (added 2026-03-21):

Each utterance's SSNR is independent (pure numpy, CPU-only), making this
stage embarrassingly parallel.  We use :class:`~concurrent.futures.ProcessPoolExecutor`
with ``chunksize=32`` to amortise IPC overhead.

Measured improvement (1665 utterances, 16 CPU cores):

    Before: ~26 s  (sequential loop)
    After:  ~2-4 s (16-worker process pool)

Worker count is ``min(os.cpu_count(), 16)`` to avoid over-subscribing
shared machines while still saturating a typical workstation.

Pipeline position
-----------------
This module is **Stage 2** of the unified evaluation pipeline
(:mod:`eval.run_all`).  It runs concurrently with TTFAT (Stage 3) since
SSNR is CPU-only and TTFAT is GPU-only -- they have zero resource
contention.

Usage::

    uv run python eval/measure_ssnr.py \\
        --experiment mimi_turkish_sample \\
        [--segment-ms 25] \\
        [--overlap-ms 10]

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torchaudio
import eval._audio_compat  # noqa: F401


from train.config_loader import load_config

logger = logging.getLogger(__name__)

# Use "spawn" so worker processes do not inherit the parent's CUDA
# driver state.  fork() + CUDA = deadlock (see bootstrap_eval.py).
_MP_CONTEXT = multiprocessing.get_context("spawn")

# Hard upper bound on worker count regardless of available cores.
_MAX_WORKERS_CAP = 16


def _select_num_workers() -> int:
    """Choose workers based on physical cores, affinity, and a hard cap.

    Decision rules:
      - 1 physical core, N logical (vCPU-only): use 1 worker.
        SSNR is numpy-only and CPU-bound; extra workers on a single
        physical core just add IPC and context-switch overhead.
      - M physical < N logical (HT): use min(physical, affinity).
      - Otherwise: use affinity count, capped at 16.

    Logs the decision and all inputs at INFO level.
    """
    logical: int = os.cpu_count() or 4
    affinity: int = logical
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    physical: int = logical
    try:
        core_ids: set[str] = set()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("core id"):
                    core_ids.add(line.split(":")[1].strip())
        if core_ids:
            physical = len(core_ids)
    except (FileNotFoundError, OSError):
        pass

    if physical == 1 and logical > 1:
        rule = (
            f"vCPU-only ({logical} logical, 1 physical): "
            "SSNR is pure numpy/CPU-bound, no benefit from time-sliced "
            "parallelism. Using 1 worker."
        )
        effective = 1
    elif physical < logical:
        effective = min(physical, affinity)
        rule = (
            f"HT machine ({physical} phys, {logical} logical, "
            f"{affinity} affinity): using {effective} workers = "
            "min(physical, affinity)."
        )
    else:
        effective = affinity
        rule = (
            f"Standard ({physical} phys, {logical} logical, "
            f"{affinity} affinity): using {effective} workers."
        )

    workers = max(1, min(effective, _MAX_WORKERS_CAP))
    logger.info(
        "SSNR worker selection: logical=%d, physical=%d, "
        "affinity=%d, cap=%d -> %d workers. %s",
        logical, physical, affinity, _MAX_WORKERS_CAP, workers, rule,
    )
    return workers


_NUM_WORKERS: int = _select_num_workers()

# Minimum signal power (mean squared amplitude) for a segment to be
# considered "active".  Segments below this threshold are excluded from
# the SSNR average to prevent silent frames from dominating the metric.
_SILENCE_THRESHOLD = 1e-8


# ---------------------------------------------------------------------------
# Core SSNR computation
# ---------------------------------------------------------------------------


def _segment_signal(
    signal: np.ndarray,
    segment_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Split a 1-D signal into overlapping frames.

    Args:
        signal: 1-D numpy array of audio samples.
        segment_samples: Length of each segment in samples.
        hop_samples: Hop size (stride) between segments in samples.

    Returns:
        2-D array of shape ``(n_frames, segment_samples)``.
    """
    n_frames = max(1, (len(signal) - segment_samples) // hop_samples + 1)
    frames = np.empty((n_frames, segment_samples), dtype=signal.dtype)
    for i in range(n_frames):
        start = i * hop_samples
        frames[i] = signal[start : start + segment_samples]
    return frames


def compute_ssnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_rate: int,
    segment_ms: float = 25.0,
    overlap_ms: float = 10.0,
) -> float:
    """Compute Segmental Signal-to-Noise Ratio for one utterance.

    Args:
        original: Original audio as a 1-D numpy array.
        reconstructed: Reconstructed audio as a 1-D numpy array.
        sample_rate: Audio sample rate in Hz.
        segment_ms: Segment length in milliseconds.
        overlap_ms: Overlap between segments in milliseconds.

    Returns:
        Mean SSNR in dB across non-silent segments.
    """
    # Align lengths.
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    noise = original - reconstructed

    segment_samples = int(segment_ms / 1000.0 * sample_rate)
    hop_samples = int((segment_ms - overlap_ms) / 1000.0 * sample_rate)
    hop_samples = max(1, hop_samples)

    sig_frames = _segment_signal(original, segment_samples, hop_samples)
    noise_frames = _segment_signal(noise, segment_samples, hop_samples)

    sig_power = np.mean(sig_frames ** 2, axis=1)
    noise_power = np.mean(noise_frames ** 2, axis=1)

    # Filter out silent segments.
    active_mask = sig_power > _SILENCE_THRESHOLD
    if not np.any(active_mask):
        return 0.0

    sig_power = sig_power[active_mask]
    noise_power = noise_power[active_mask]

    eps = 1e-10
    segment_snr = 10.0 * np.log10(sig_power / (noise_power + eps))

    return float(np.mean(segment_snr))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _compute_ssnr_for_entry(
    args: Tuple[str, str, int, float, float],
) -> float:
    """Process-pool worker: compute SSNR for a single utterance pair.

    This function is the unit of work dispatched to each process in the
    :class:`~concurrent.futures.ProcessPoolExecutor`.  It accepts a
    single tuple (rather than keyword arguments) because
    :meth:`~concurrent.futures.Executor.map` requires a single-argument
    callable.

    Args:
        args: A 5-tuple of ``(ref_path, deg_path, sample_rate,
            segment_ms, overlap_ms)``.

    Returns:
        Mean SSNR in dB for the utterance.
    """
    ref_path, deg_path, sample_rate, segment_ms, overlap_ms = args

    ref_wav, _ = torchaudio.load(ref_path)
    deg_wav, _ = torchaudio.load(deg_path)

    # Ensure mono before converting to numpy.
    if ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(dim=0, keepdim=True)
    if deg_wav.shape[0] > 1:
        deg_wav = deg_wav.mean(dim=0, keepdim=True)

    return compute_ssnr(
        ref_wav.squeeze(0).numpy(),
        deg_wav.squeeze(0).numpy(),
        sample_rate,
        segment_ms=segment_ms,
        overlap_ms=overlap_ms,
    )


def run(
    experiment: str,
    segment_ms: float = 25.0,
    overlap_ms: float = 10.0,
) -> Dict[str, Any]:
    """Run SSNR measurement across all reconstructed utterances.

    Uses a process pool for parallel computation across CPU cores.

    Args:
        experiment: Experiment name (maps to a config under
            ``configs/experiments/``).
        segment_ms: Segment length in milliseconds.
        overlap_ms: Overlap between segments in milliseconds.

    Returns:
        Dict with SSNR statistics.

    Raises:
        FileNotFoundError: If the reconstruction manifest is missing.
    """
    config_path = Path("configs/experiments") / f"{experiment}.yaml"
    config = load_config(str(config_path))
    sample_rate = int(config["codec"]["sample_rate"])

    recon_manifest_path = (
        Path(config["output_dir"]) / "reconstructed" / "test" / "manifest.json"
    )
    if not recon_manifest_path.exists():
        raise FileNotFoundError(
            f"Reconstruction manifest not found: {recon_manifest_path}. "
            "Run eval/reconstruct.py first."
        )

    with open(recon_manifest_path, "r", encoding="utf-8") as fh:
        manifest: List[Dict[str, Any]] = json.load(fh)

    work_items = [
        (
            entry["original_path"],
            entry["reconstructed_path"],
            sample_rate,
            segment_ms,
            overlap_ms,
        )
        for entry in manifest
    ]

    total = len(work_items)

    # chunksize=32 is appropriate for SSNR because each item is very
    # lightweight (~1ms of numpy ops).  Larger chunks amortise the IPC
    # overhead of sending work items and results between processes.
    # With chunksize=1 the per-item IPC cost would dominate.
    chunksize = 32

    logger.info(
        "SSNR pool config: workers=%d, chunksize=%d, "
        "mp_start_method='%s', total_items=%d. "
        "Each utterance is ~1ms of numpy segment-wise SNR. "
        "Using 'spawn' to avoid CUDA fork deadlock.",
        _NUM_WORKERS, chunksize, _MP_CONTEXT.get_start_method(), total,
    )

    import time as _time

    ssnr_values: List[float] = []
    t_start = _time.monotonic()
    log_interval = max(1, min(total // 10, 100))

    with ProcessPoolExecutor(
        max_workers=_NUM_WORKERS, mp_context=_MP_CONTEXT,
    ) as pool:
        for val in pool.map(_compute_ssnr_for_entry, work_items, chunksize=chunksize):
            ssnr_values.append(val)
            done = len(ssnr_values)
            if done % log_interval == 0 or done == total:
                elapsed = _time.monotonic() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                logger.info(
                    "SSNR progress: %d/%d (%.0f%%) "
                    "[%.1fs elapsed, %.1f utt/s]",
                    done, total, 100 * done / total, elapsed, rate,
                )

    ssnr_arr = np.array(ssnr_values, dtype=np.float64)
    stats: Dict[str, Any] = {
        "ssnr_mean": round(float(np.mean(ssnr_arr)), 4),
        "ssnr_std": round(float(np.std(ssnr_arr, ddof=1)), 4),
        "ssnr_min": round(float(np.min(ssnr_arr)), 4),
        "ssnr_max": round(float(np.max(ssnr_arr)), 4),
        "n_utterances": len(manifest),
        "segment_ms": segment_ms,
        "overlap_ms": overlap_ms,
        "experiment": experiment,
    }

    print(
        f"\n{'═' * 60}\n"
        f"Segmental SNR: {experiment}\n"
        f"{'─' * 60}\n"
        f"  Mean SSNR  : {stats['ssnr_mean']:.2f} dB\n"
        f"  Std        : {stats['ssnr_std']:.2f} dB\n"
        f"  Min        : {stats['ssnr_min']:.2f} dB\n"
        f"  Max        : {stats['ssnr_max']:.2f} dB\n"
        f"  Utterances : {stats['n_utterances']}\n"
        f"{'═' * 60}"
    )

    # Save results.
    out_path = Path(f"results/{experiment}_ssnr.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Results saved to %s", out_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and measure SSNR."""
    parser = argparse.ArgumentParser(
        description="Measure Segmental Signal-to-Noise Ratio.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g. mimi_turkish_sample).",
    )
    parser.add_argument(
        "--segment-ms",
        type=float,
        default=25.0,
        help="Segment length in ms (default: 25).",
    )
    parser.add_argument(
        "--overlap-ms",
        type=float,
        default=10.0,
        help="Overlap between segments in ms (default: 10).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(
        experiment=args.experiment,
        segment_ms=args.segment_ms,
        overlap_ms=args.overlap_ms,
    )


if __name__ == "__main__":
    main()
