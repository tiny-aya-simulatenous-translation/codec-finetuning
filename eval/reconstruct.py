"""Reconstruct audio through a trained codec for evaluation.

Loads a fine-tuned (or pretrained) codec checkpoint, encodes each test
utterance into discrete tokens, decodes back to audio, and saves the
reconstructed waveforms alongside a manifest for downstream metrics.

Supports all three codecs: Mimi, DualCodec, and Kanade.

Usage::

    uv run python eval/reconstruct.py \
        --config configs/experiments/mimi_turkish_sample.yaml \
        [--checkpoint outputs/mimi_turkish_sample/best.pt] \
        [--split test] \
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
import soundfile as sf
import torch
import torchaudio

from train.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Codec loading helpers
# ---------------------------------------------------------------------------


def _load_mimi(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> torch.nn.Module:
    """Load the Mimi codec model from pretrained or fine-tuned checkpoint.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint ``.pt`` file.
        use_ema: Whether to load EMA weights from the checkpoint.
        device: Target device (``"cuda"`` or ``"cpu"``).

    Returns:
        The Mimi model ready for inference.
    """
    from transformers import MimiModel

    model = MimiModel.from_pretrained(config["codec"]["pretrained"])

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if use_ema and "ema_state_dict" in ckpt:
            logger.info("Loading EMA weights from checkpoint.")
            model.load_state_dict(ckpt["ema_state_dict"])
        elif "model_state_dict" in ckpt:
            logger.info("Loading model weights from checkpoint.")
            model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    model.eval()
    return model


def _load_dualcodec(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    """Load the DualCodec model.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        use_ema: Whether to load EMA weights from the checkpoint.
        device: Target device.

    Returns:
        The DualCodec model ready for inference.
    """
    import dualcodec

    model = dualcodec.load_model(config["codec"]["pretrained"])

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        has_ema = use_ema and "ema_state_dict" in ckpt
        state_key = "ema_state_dict" if has_ema else "model_state_dict"
        if state_key in ckpt:
            logger.info("Loading %s from checkpoint.", state_key)
            model.load_state_dict(ckpt[state_key])

    model = model.to(device)
    model.eval()
    return model


def _load_kanade(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    """Load the Kanade tokenizer model.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        use_ema: Whether to load EMA weights from the checkpoint.
        device: Target device.

    Returns:
        The Kanade model ready for inference.
    """
    import kanade_tokenizer

    model = kanade_tokenizer.load_model(config["codec"]["pretrained"])

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        has_ema = use_ema and "ema_state_dict" in ckpt
        state_key = "ema_state_dict" if has_ema else "model_state_dict"
        if state_key in ckpt:
            logger.info("Loading %s from checkpoint.", state_key)
            model.load_state_dict(ckpt[state_key])

    model = model.to(device)
    model.eval()
    return model


def load_model(
    config: Dict[str, Any],
    checkpoint: Optional[str] = None,
    use_ema: bool = False,
    device: Optional[torch.device] = None,
) -> Any:
    """Load the appropriate codec model based on config.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        use_ema: Whether to load EMA weights.
        device: Target device. Defaults to CUDA if available.

    Returns:
        The loaded codec model.

    Raises:
        ValueError: If the codec name in config is not supported.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codec_name = config["codec"]["name"].lower()
    loaders = {
        "mimi": _load_mimi,
        "dualcodec": _load_dualcodec,
        "kanade": _load_kanade,
    }

    if codec_name not in loaders:
        raise ValueError(
            f"Unsupported codec '{codec_name}'. "
            f"Supported: {', '.join(loaders)}"
        )

    return loaders[codec_name](config, checkpoint, use_ema, device)


# ---------------------------------------------------------------------------
# Encode / decode dispatch
# ---------------------------------------------------------------------------


def _encode_decode_mimi(
    model: torch.nn.Module,
    waveform: torch.Tensor,
) -> torch.Tensor:
    """Encode and decode a waveform using the Mimi codec.

    Args:
        model: The loaded Mimi model.
        waveform: Input audio tensor of shape ``(1, 1, samples)``.

    Returns:
        Reconstructed audio tensor of shape ``(1, 1, samples)``.
    """
    tokens = model.encode(waveform)
    audio_codes = tokens.audio_codes
    reconstructed = model.decode(audio_codes)
    return reconstructed.audio_values


def _encode_decode_dualcodec(
    model: Any,
    waveform: torch.Tensor,
) -> torch.Tensor:
    """Encode and decode a waveform using DualCodec.

    Args:
        model: The loaded DualCodec model.
        waveform: Input audio tensor of shape ``(1, 1, samples)``.

    Returns:
        Reconstructed audio tensor of shape ``(1, 1, samples)``.
    """
    tokens = model.encode(waveform)
    reconstructed = model.decode(tokens)
    return reconstructed


def _encode_decode_kanade(
    model: Any,
    waveform: torch.Tensor,
) -> torch.Tensor:
    """Encode and decode a waveform using Kanade tokenizer.

    Args:
        model: The loaded Kanade model.
        waveform: Input audio tensor of shape ``(1, 1, samples)``.

    Returns:
        Reconstructed audio tensor of shape ``(1, 1, samples)``.
    """
    tokens = model.encode(waveform)
    reconstructed = model.decode(tokens)
    return reconstructed


def encode_decode(
    model: Any,
    waveform: torch.Tensor,
    codec_name: str,
) -> torch.Tensor:
    """Dispatch encode-decode to the correct codec implementation.

    Args:
        model: The loaded codec model.
        waveform: Input audio tensor of shape ``(1, 1, samples)``.
        codec_name: One of ``"mimi"``, ``"dualcodec"``, ``"kanade"``.

    Returns:
        Reconstructed audio tensor.

    Raises:
        ValueError: If *codec_name* is not recognised.
    """
    dispatch = {
        "mimi": _encode_decode_mimi,
        "dualcodec": _encode_decode_dualcodec,
        "kanade": _encode_decode_kanade,
    }
    if codec_name not in dispatch:
        raise ValueError(f"Unknown codec: {codec_name}")
    return dispatch[codec_name](model, waveform)


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


def reconstruct(
    config: Dict[str, Any],
    checkpoint: Optional[str] = None,
    split: str = "test",
    use_ema: bool = False,
) -> Path:
    """Run the full reconstruction pipeline for a dataset split.

    Args:
        config: Full experiment config dict.
        checkpoint: Optional path to a fine-tuned checkpoint.
        split: Dataset split to reconstruct (``"test"``, ``"val"``).
        use_ema: Whether to load EMA weights.

    Returns:
        Path to the output manifest JSON.

    Raises:
        FileNotFoundError: If the input manifest is missing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_name = config["codec"]["name"].lower()
    sample_rate = int(config["codec"]["sample_rate"])
    latency_ms = float(config["codec"].get("latency_ms", 0))

    # Load model.
    model = load_model(config, checkpoint, use_ema, device)

    # Input manifest.
    data_dir = Path(config["dataset"]["local_dir"])
    manifest_path = data_dir / split / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest: List[Dict[str, Any]] = json.load(fh)

    # Output directory.
    output_dir = Path(config["output_dir"]) / "reconstructed" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total_duration = 0.0
    start_time = time.monotonic()

    for entry in manifest:
        utt_id = entry.get("id", Path(entry["audio_path"]).stem)
        audio_path = entry["audio_path"]

        # Load audio.
        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        duration = waveform.shape[-1] / sample_rate
        total_duration += duration

        # Encode + decode.
        waveform_input = waveform.unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            reconstructed = encode_decode(model, waveform_input, codec_name)

        reconstructed = reconstructed.squeeze(0).cpu()

        # Latency alignment.
        waveform, reconstructed = _align_latency(
            waveform, reconstructed, latency_ms, sample_rate
        )

        # Save reconstructed audio.
        out_path = output_dir / f"{utt_id}.wav"
        sf.write(str(out_path), reconstructed.squeeze(0).numpy(), sample_rate)

        results.append({
            "id": utt_id,
            "original_path": str(audio_path),
            "reconstructed_path": str(out_path),
            "duration": round(duration, 4),
        })

    # Save output manifest.
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
