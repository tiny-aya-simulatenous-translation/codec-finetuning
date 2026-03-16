"""Audio data augmentation utilities with preset configurations.

Provides a pipeline of waveform-level augmentations designed for codec
fine-tuning on low-resource speech data.  Three built-in presets
(``"none"``, ``"light"``, ``"heavy"``) cover common training scenarios;
``"custom"`` lets callers override every knob.

Usage::

    from train.utils.augmentation import AugmentationConfig, augment_waveform

    cfg = AugmentationConfig(preset="light")
    augmented = augment_waveform(waveform, sr=24_000, config=cfg)

Dependencies:
    torch, torchaudio

License: MIT
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torchaudio.functional as AF
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Declarative augmentation configuration.

    Attributes:
        preset: One of ``"none"``, ``"light"``, ``"heavy"``, ``"custom"``.
        speed_perturb: Speed perturbation range as ``(low, high)`` factor.
            ``None`` disables.
        pitch_shift: Pitch-shift range in semitones ``(low, high)``.
            ``None`` disables.
        noise_snr: Additive Gaussian noise SNR range in dB ``(low, high)``.
            ``None`` disables.
        gain_db: Random gain jitter range in dB ``(low, high)``.
            ``None`` disables.
        chunk_reverse_prob: Probability of reversing a random chunk.
            ``0.0`` disables.
    """

    preset: str = "none"
    speed_perturb: Tuple[float, float] | None = None
    pitch_shift: Tuple[float, float] | None = None
    noise_snr: Tuple[float, float] | None = None
    gain_db: Tuple[float, float] | None = None
    chunk_reverse_prob: float = 0.0


# Preset look-up: each value is a dict of field overrides.
_PRESETS: dict[str, dict] = {
    "none": {},
    "light": {
        "speed_perturb": (0.95, 1.05),
        "noise_snr": (30.0, 50.0),
        "gain_db": (-2.0, 2.0),
        "chunk_reverse_prob": 0.0,
    },
    "heavy": {
        "speed_perturb": (0.85, 1.15),
        "pitch_shift": (-2.0, 2.0),
        "noise_snr": (10.0, 30.0),
        "gain_db": (-6.0, 6.0),
        "chunk_reverse_prob": 0.15,
    },
}


def resolve_preset(config: AugmentationConfig) -> AugmentationConfig:
    """Fill unset fields from the named preset.

    If ``config.preset`` is ``"custom"``, the config is returned as-is so
    that callers retain full control.

    Args:
        config: Augmentation configuration, potentially with ``None`` fields.

    Returns:
        A new ``AugmentationConfig`` with defaults filled in from the preset.

    Raises:
        ValueError: If ``config.preset`` is not a recognised name.

    Example::

        >>> cfg = resolve_preset(AugmentationConfig(preset="light"))
        >>> cfg.speed_perturb
        (0.95, 1.05)
    """
    if config.preset == "custom":
        return config

    if config.preset not in _PRESETS:
        raise ValueError(
            f"Unknown preset {config.preset!r}. "
            f"Choose from {list(_PRESETS.keys()) + ['custom']}."
        )

    overrides = _PRESETS[config.preset]
    return AugmentationConfig(preset=config.preset, **overrides)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def augment_waveform(
    waveform: Tensor,
    sr: int,
    config: AugmentationConfig,
    rng: random.Random | None = None,
) -> Tensor:
    """Apply a chain of waveform augmentations.

    Augmentations are applied in a fixed order so that their interactions
    remain predictable (e.g. speed perturbation before pitch shifting avoids
    double-resampling artefacts).

    Args:
        waveform: Audio tensor of shape ``(channels, samples)`` or
            ``(samples,)``.
        sr: Sample rate in Hz.
        config: Resolved augmentation configuration.
        rng: Optional ``random.Random`` instance for reproducibility.
            Falls back to the module-level RNG if ``None``.

    Returns:
        Augmented waveform with the same shape and device as the input.

    Example::

        >>> wav = torch.randn(1, 24000)
        >>> cfg = resolve_preset(AugmentationConfig(preset="light"))
        >>> out = augment_waveform(wav, 24000, cfg)
        >>> out.shape
        torch.Size([1, 24000])
    """
    if rng is None:
        rng = random.Random()

    config = resolve_preset(config)
    orig_len = waveform.shape[-1]

    # 1. Speed perturbation — resample to a perturbed rate then back to the
    #    original rate, effectively stretching or compressing in time.
    if config.speed_perturb is not None:
        lo, hi = config.speed_perturb
        factor = rng.uniform(lo, hi)
        # Perturbed rate: higher factor → faster speech → fewer samples.
        perturbed_sr = int(round(sr * factor))
        waveform = AF.resample(waveform, sr, perturbed_sr)
        waveform = AF.resample(waveform, perturbed_sr, sr)

    # 2. Pitch shift — shift pitch by a random number of semitones.  Uses
    #    resampling trick: resample to shift, then time-stretch back.
    if config.pitch_shift is not None:
        lo, hi = config.pitch_shift
        semitones = rng.uniform(lo, hi)
        # Ratio converts semitones to frequency multiplier (12-TET).
        ratio = 2.0 ** (semitones / 12.0)
        shifted_sr = int(round(sr * ratio))
        waveform = AF.resample(waveform, sr, shifted_sr)
        waveform = AF.resample(waveform, shifted_sr, sr)

    # 3. Additive Gaussian noise — simulates recording noise.  SNR is
    #    randomised so the model sees a range of noise conditions.
    if config.noise_snr is not None:
        lo, hi = config.noise_snr
        snr_db = rng.uniform(lo, hi)
        # Convert SNR (dB) to linear scale for mixing.
        signal_power = waveform.square().mean().clamp(min=1e-10)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        waveform = waveform + noise

    # 4. Gain jitter — random volume change to make the model invariant to
    #    recording levels.
    if config.gain_db is not None:
        lo, hi = config.gain_db
        gain_db = rng.uniform(lo, hi)
        waveform = waveform * (10.0 ** (gain_db / 20.0))

    # 5. Chunk reversal — reverse a random 10-30 % contiguous chunk.  This
    #    acts as a mild temporal perturbation that forces the model to rely
    #    on local rather than global temporal patterns.
    if config.chunk_reverse_prob > 0.0 and rng.random() < config.chunk_reverse_prob:
        n = waveform.shape[-1]
        chunk_frac = rng.uniform(0.10, 0.30)
        chunk_len = max(1, int(n * chunk_frac))
        start = rng.randint(0, max(0, n - chunk_len))
        waveform[..., start : start + chunk_len] = waveform[
            ..., start : start + chunk_len
        ].flip(-1)

    # Pad or trim to original length so downstream code sees a fixed size.
    if waveform.shape[-1] < orig_len:
        pad = orig_len - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.shape[-1] > orig_len:
        waveform = waveform[..., :orig_len]

    # Safety: if all augmentations produced silence, return the un-augmented
    # input to avoid training on zeros.
    if waveform.abs().max() < 1e-6:
        return torch.randn_like(waveform) * 1e-4  # tiny noise as fallback

    return waveform
