"""Time to First Audio Token (TTFAT) measurement.

Measures the latency from audio input to the first encoded token by
benchmarking the codec encoder with CUDA synchronisation barriers.

Pipeline position
-----------------
This module is **Stage 3** of the unified evaluation pipeline
(:mod:`eval.run_all`).  It can also be run standalone.

Methodology
-----------
1. A single codec frame (``latency_ms`` worth of audio) is loaded from a
   random test utterance.
2. The frame is encoded through the codec's ``encode()`` method.
3. Wall-clock time is measured with ``time.perf_counter_ns()`` bracketed
   by ``torch.cuda.synchronize()`` calls to include all GPU work.
4. Warmup passes are excluded from the timing to avoid cold-start JIT
   and memory-allocation overhead.
5. Statistics (mean, std, percentiles) are computed over *n_runs* timed
   passes.

Usage::

    uv run python eval/measure_ttfat.py \\
        --config configs/experiments/mimi_turkish_sample.yaml \\
        [--n-runs 50] \\
        [--warmup 5]

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
import eval._audio_compat  # noqa: F401


from eval.reconstruct import load_model
from train.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def measure_ttfat(
    config: Dict[str, Any],
    checkpoint: Optional[str] = None,
    n_runs: int = 50,
    warmup: int = 5,
) -> Dict[str, Any]:
    """Measure Time to First Audio Token latency.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        n_runs: Number of timed encoding runs.
        warmup: Number of warmup forward passes before measurement.

    Returns:
        Dict with timing statistics in both nanoseconds and milliseconds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    sample_rate = int(config["codec"]["sample_rate"])
    latency_ms = float(config["codec"].get("latency_ms", 80))
    codec_name = config["codec"]["name"].lower()

    # Number of audio samples in one codec frame.  For example, at
    # 24 kHz with 80 ms latency this equals 1920 samples -- the minimum
    # input size the encoder needs to produce one discrete token.
    frame_samples = int(latency_ms / 1000.0 * sample_rate)

    # Load model.
    model = load_model(config, checkpoint, use_ema=False, device=device)

    # Load test manifest for random utterance selection.
    data_dir = Path(config["dataset"]["local_dir"])
    manifest_path = data_dir / "test" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest: List[Dict[str, Any]] = json.load(fh)

    def _load_random_frame() -> torch.Tensor:
        """Load a random utterance and extract one codec frame.

        Returns:
            A tensor of shape ``(1, 1, frame_samples)`` on *device*,
            containing exactly one codec frame of audio.
        """
        entry = random.choice(manifest)
        audio_path = entry["audio_path"]
        if not Path(audio_path).is_absolute():
            audio_path = str(data_dir / audio_path)
        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, sample_rate
            )
        # Down-mix to mono if multi-channel.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Slice exactly one codec frame from the beginning.
        waveform = waveform[:, :frame_samples]
        # Zero-pad short utterances so the encoder always gets a full frame.
        if waveform.shape[-1] < frame_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, frame_samples - waveform.shape[-1])
            )
        # Add batch dimension: (1, 1, frame_samples).
        return waveform.unsqueeze(0).to(device)

    # Warmup passes.
    logger.info("Running %d warmup passes...", warmup)
    for _ in range(warmup):
        frame = _load_random_frame()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            model.encode(frame) if hasattr(model, "encode") else model(frame)
        if use_cuda:
            torch.cuda.synchronize()

    # Timed runs.
    logger.info("Running %d timed encoding passes...", n_runs)
    times_ns: List[int] = []

    for _ in range(n_runs):
        frame = _load_random_frame()

        if use_cuda:
            torch.cuda.synchronize()

        t_start = time.perf_counter_ns()

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            model.encode(frame) if hasattr(model, "encode") else model(frame)

        if use_cuda:
            torch.cuda.synchronize()

        t_end = time.perf_counter_ns()
        times_ns.append(t_end - t_start)

    # Convert nanoseconds to milliseconds for human-readable reporting.
    times_ms = np.array(times_ns, dtype=np.float64) / 1e6
    stats: Dict[str, Any] = {
        "mean_ms": round(float(np.mean(times_ms)), 3),
        "std_ms": round(float(np.std(times_ms, ddof=1)), 3),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(times_ms, 99)), 3),
        "min_ms": round(float(np.min(times_ms)), 3),
        "max_ms": round(float(np.max(times_ms)), 3),
        "n_runs": n_runs,
        "warmup": warmup,
        "codec": codec_name,
        "frame_samples": frame_samples,
        "latency_ms": latency_ms,
    }

    # Print results.
    print(
        f"\n{'═' * 60}\n"
        f"Time to First Audio Token (TTFAT)\n"
        f"{'─' * 60}\n"
        f"  Codec      : {codec_name}\n"
        f"  Runs       : {n_runs} (warmup: {warmup})\n"
        f"  Mean       : {stats['mean_ms']:.3f} ms\n"
        f"  Std        : {stats['std_ms']:.3f} ms\n"
        f"  P50        : {stats['p50_ms']:.3f} ms\n"
        f"  P95        : {stats['p95_ms']:.3f} ms\n"
        f"  P99        : {stats['p99_ms']:.3f} ms\n"
        f"{'═' * 60}"
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and measure TTFAT."""
    parser = argparse.ArgumentParser(
        description="Measure Time to First Audio Token latency.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Number of timed runs (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup forward passes (default: 5).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    stats = measure_ttfat(
        config,
        checkpoint=args.checkpoint,
        n_runs=args.n_runs,
        warmup=args.warmup,
    )

    # Save results.
    experiment = Path(args.config).stem
    out_path = Path(f"results/{experiment}_ttfat.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
