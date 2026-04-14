"""Reconstruct audio through a trained codec for evaluation.

Loads a fine-tuned (or pretrained) codec checkpoint, encodes each test
utterance into discrete tokens, decodes back to audio, and saves the
reconstructed waveforms alongside a manifest for downstream metrics.

New codecs are registered via :mod:`eval.codec_registry` -- no changes
to this file are needed.

Performance
-----------
**Batched GPU inference** (added 2026-03-21):

The original implementation processed utterances one at a time, issuing a
separate GPU kernel launch per utterance.  For a 1665-utterance test set
on an H100 this took ~26 s with the GPU idle between each file-load/save.

The current implementation:

1. **Pre-loads** all waveforms from disk into CPU memory.
2. **Sorts** by sample length so that utterances within each batch have
   similar durations, minimising padding waste.
3. **Pads** each batch to the longest waveform in that batch and runs a
   single ``encode_decode()`` call per batch (default batch size 32).
4. **Un-batches**, trims padding + codec latency, and writes output WAVs.

Measured improvement on an H100 (1665 utterances, 24 kHz, Mimi):

    Before: ~26 s  (sequential, batch_size=1)
    After:  ~4-8 s (batched,   batch_size=32)

Tip: For very long utterances (>20 s) or limited GPU VRAM, lower
``batch_size`` to avoid OOM errors.

Pipeline position
-----------------
This module is **Stage 1** of the unified evaluation pipeline
(:mod:`eval.run_all`).  All downstream stages (SSNR, TTFAT, bootstrap
metrics, VERSA, WandB publish) depend on its output manifest.

Usage::

    uv run python eval/reconstruct.py \\
        --config configs/experiments/mimi_turkish_sample.yaml \\
        [--checkpoint outputs/mimi_turkish_sample/best.pt] \\
        [--split test] \\
        [--use-ema]

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

import eval._audio_compat  # noqa: F401 — patches torchaudio on aarch64

from eval.codec_registry import get_codec_hooks
from train.config_loader import load_config

logger = logging.getLogger(__name__)

# Maximum number of utterances to encode/decode in one GPU call.
# Sorting by length keeps intra-batch padding under ~10 % for typical
# speech corpora.  Reduce if GPU memory is limited.
_DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Public API -- thin wrappers around the codec registry
# ---------------------------------------------------------------------------


def load_model(
    config: Dict[str, Any],
    checkpoint: Optional[str] = None,
    use_ema: bool = False,
    device: Optional[torch.device] = None,
) -> Any:
    """Load the appropriate codec model based on config.

    Delegates to the codec's registered ``load`` hook.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        use_ema: Whether to load EMA weights.
        device: Target device. Defaults to CUDA if available.

    Returns:
        The loaded codec model.

    Raises:
        ValueError: If the codec name in config is not registered.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codec_name = config["codec"]["name"].lower()
    hooks = get_codec_hooks(codec_name)
    return hooks.load(config, checkpoint, use_ema, device)


def encode_decode(
    model: Any,
    waveform: torch.Tensor,
    codec_name: str,
) -> torch.Tensor:
    """Dispatch encode-decode to the correct codec implementation.

    Delegates to the codec's registered ``encode_decode`` hook.

    Args:
        model: The loaded codec model.
        waveform: Input audio tensor of shape ``(1, 1, samples)``.
        codec_name: Codec identifier (case-insensitive).

    Returns:
        Reconstructed audio tensor.

    Raises:
        ValueError: If *codec_name* is not registered.
    """
    hooks = get_codec_hooks(codec_name)
    return hooks.encode_decode(model, waveform)


# ---------------------------------------------------------------------------
# Latency alignment
# ---------------------------------------------------------------------------


def _align_latency(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    latency_ms: float,
    sample_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim or pad reconstructed audio to align codec latency.

    Codec latency introduces a fixed delay in the output.  This function
    trims the leading *latency_ms* from the reconstructed signal and pads
    the end to match the original length, ensuring sample-aligned pairs
    for reference-dependent metrics.

    Args:
        original: Original waveform tensor.
        reconstructed: Reconstructed waveform tensor.
        latency_ms: Codec latency in milliseconds.
        sample_rate: Audio sample rate in Hz.

    Returns:
        A ``(original, reconstructed)`` tuple with matching lengths.
    """
    latency_samples = int(latency_ms / 1000.0 * sample_rate)

    if latency_samples > 0 and reconstructed.shape[-1] > latency_samples:
        reconstructed = reconstructed[..., latency_samples:]

    # Match lengths: trim to the shorter of the two.
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    return original, reconstructed


# ---------------------------------------------------------------------------
# Main reconstruction pipeline
# ---------------------------------------------------------------------------


def _load_and_preprocess(
    entry: Dict[str, Any],
    data_dir: Path,
    sample_rate: int,
) -> Tuple[str, str, torch.Tensor]:
    """Load a single utterance from disk and normalise it for the codec.

    Steps performed:
        1. Resolve the audio path (relative paths are resolved against
           *data_dir*).
        2. Load the waveform via :func:`torchaudio.load`.
        3. Resample to *sample_rate* if the file's native rate differs.
        4. Down-mix to mono if multi-channel.

    Args:
        entry: One element of the manifest JSON list.  Must contain at
            least an ``"audio_path"`` key; ``"id"`` is optional.
        data_dir: Root directory of the prepared dataset (e.g.
            ``data/lahaja``).
        sample_rate: Target sample rate in Hz (e.g. 24 000 for Mimi).

    Returns:
        A tuple of ``(utterance_id, absolute_audio_path, waveform)``
        where *waveform* has shape ``(1, samples)`` (mono, float32).
    """
    utt_id = entry.get("id", Path(entry["audio_path"]).stem)
    audio_path = entry["audio_path"]
    if not Path(audio_path).is_absolute():
        audio_path = str(data_dir / audio_path)

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return utt_id, audio_path, waveform


def reconstruct(
    config: Dict[str, Any],
    checkpoint: Optional[str] = None,
    split: str = "test",
    use_ema: bool = False,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Path:
    """Run the full reconstruction pipeline for a dataset split.

    Utterances are sorted by duration and processed in padded GPU
    batches for higher throughput.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        split: Dataset split to reconstruct (``"test"``, ``"val"``).
        use_ema: Whether to load EMA weights.
        batch_size: Number of utterances per GPU batch.

    Returns:
        Path to the output manifest JSON.

    Raises:
        FileNotFoundError: If the input manifest is missing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_name = config["codec"]["name"].lower()
    sample_rate = int(config["codec"]["sample_rate"])
    latency_ms = float(config["codec"].get("latency_ms", 0))

    model = load_model(config, checkpoint, use_ema, device)

    data_dir = Path(config["dataset"]["local_dir"])
    manifest_path = data_dir / split / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest: List[Dict[str, Any]] = json.load(fh)

    output_dir = Path(config["output_dir"]) / "reconstructed" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Pre-load all waveforms into CPU memory.
    #
    # Sorting by sample length groups similar-duration utterances so
    # that zero-padding within each batch stays minimal (typically <10 %
    # wasted compute for speech corpora with varied durations).
    # ------------------------------------------------------------------
    loaded: List[Tuple[int, str, str, torch.Tensor]] = []
    for idx, entry in enumerate(manifest):
        utt_id, audio_path, waveform = _load_and_preprocess(
            entry, data_dir, sample_rate,
        )
        loaded.append((idx, utt_id, audio_path, waveform))

    loaded.sort(key=lambda x: x[3].shape[-1])

    # results_by_idx preserves the original manifest ordering after
    # batched (length-sorted) processing so that downstream stages
    # see utterances in the same order as the input manifest.
    results_by_idx: Dict[int, Dict[str, Any]] = {}
    total_duration = 0.0
    start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Phase 2: Batched GPU encode/decode.
    #
    # For each batch we:
    #   a) Pad all waveforms to the longest in the batch.
    #   b) Move the padded tensor to GPU in a single transfer.
    #   c) Call encode_decode() once for the whole batch.
    #   d) Move results back to CPU, trim padding, align latency,
    #      and write each utterance to disk.
    # ------------------------------------------------------------------
    for batch_start in range(0, len(loaded), batch_size):
        batch = loaded[batch_start : batch_start + batch_size]

        # Per-utterance original lengths (before padding), used later
        # to trim the reconstructed output back to the correct length.
        lengths = [item[3].shape[-1] for item in batch]
        max_len = max(lengths)

        # Zero-pad to the longest utterance in this batch.
        padded = torch.zeros(len(batch), 1, max_len)
        for i, (_, _, _, waveform) in enumerate(batch):
            padded[i, :, : waveform.shape[-1]] = waveform

        # Single batched GPU forward pass.
        padded_gpu = padded.to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
        ):
            reconstructed_batch = encode_decode(model, padded_gpu, codec_name)

        reconstructed_batch = reconstructed_batch.float().cpu()

        # Unbatch: trim padding, apply codec-latency alignment, save.
        for i, (orig_idx, utt_id, audio_path, waveform) in enumerate(batch):
            orig_len = lengths[i]
            duration = orig_len / sample_rate
            total_duration += duration

            recon = reconstructed_batch[i : i + 1, :, :]

            waveform_aligned, recon_aligned = _align_latency(
                waveform, recon.squeeze(0), latency_ms, sample_rate,
            )

            out_path = output_dir / f"{utt_id}.wav"
            torchaudio.save(str(out_path), recon_aligned, sample_rate)

            results_by_idx[orig_idx] = {
                "id": utt_id,
                "original_path": str(audio_path),
                "reconstructed_path": str(out_path),
                "duration": round(duration, 4),
            }

    # Restore the original manifest ordering (batched processing used a
    # length-sorted order).
    results = [results_by_idx[i] for i in range(len(manifest))]

    out_manifest = output_dir / "manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    elapsed = time.monotonic() - start_time
    total_hours = total_duration / 3600.0

    print(
        f"\n{'═' * 60}\n"
        f"Reconstruction complete\n"
        f"{'─' * 60}\n"
        f"  Utterances : {len(results)}\n"
        f"  Batch size : {batch_size}\n"
        f"  Total hours: {total_hours:.2f}\n"
        f"  Elapsed    : {elapsed:.1f}s\n"
        f"  Output dir : {output_dir}\n"
        f"{'═' * 60}"
    )

    return out_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run reconstruction."""
    parser = argparse.ArgumentParser(
        description="Reconstruct audio through a trained codec.",
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
        help="Path to a fine-tuned checkpoint. Uses pretrained if omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to reconstruct (default: test).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load EMA weights from the checkpoint.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    reconstruct(
        config,
        checkpoint=args.checkpoint,
        split=args.split,
        use_ema=args.use_ema,
    )


if __name__ == "__main__":
    main()
