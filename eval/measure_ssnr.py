"""Segmental Signal-to-Noise Ratio (SSNR) measurement.

Computes SSNR between original and reconstructed audio by segmenting
waveforms into overlapping frames and averaging per-segment SNR values,
excluding silent segments.

Usage::

    uv run python eval/measure_ssnr.py \
        --experiment mimi_turkish_sample \
        [--segment-ms 25] \
        [--overlap-ms 10]

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from train.config_loader import load_config

logger = logging.getLogger(__name__)

# Minimum signal power threshold to exclude silent segments.
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


def run(
    experiment: str,
    segment_ms: float = 25.0,
    overlap_ms: float = 10.0,
) -> Dict[str, Any]:
    """Run SSNR measurement across all reconstructed utterances.

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

    logger.info("Computing SSNR for %d utterances...", len(manifest))

    ssnr_values: List[float] = []
    for entry in manifest:
        ref, _ = sf.read(entry["original_path"])
        deg, _ = sf.read(entry["reconstructed_path"])

        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        if deg.ndim > 1:
            deg = deg.mean(axis=1)

        ssnr = compute_ssnr(
            ref, deg, sample_rate,
            segment_ms=segment_ms,
            overlap_ms=overlap_ms,
        )
        ssnr_values.append(ssnr)

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
