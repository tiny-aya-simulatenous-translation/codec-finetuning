"""Audio loading compatibility for platforms without torchcodec (aarch64).

torchaudio >=2.9 defaults to torchcodec for load/save, which is unavailable
on aarch64.  Import this module early to monkey-patch torchaudio with
soundfile-based fallbacks when torchcodec is missing.

Usage::

    import eval._audio_compat  # noqa: F401 — patches torchaudio
"""

from __future__ import annotations

import torch
import torchaudio

_NEEDS_PATCH = False
try:
    torchaudio.load.__module__  # just probe — real check is below
    # Test if torchcodec actually works
    import torchcodec  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _NEEDS_PATCH = True


def _load_soundfile(filepath, **kwargs):
    """Load audio via soundfile, returning (waveform, sample_rate)."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    # soundfile returns (samples, channels), torch expects (channels, samples)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, sr


def _save_soundfile(filepath, waveform, sample_rate, **kwargs):
    """Save audio via soundfile."""
    import soundfile as sf

    # torch (channels, samples) -> soundfile (samples, channels)
    data = waveform.cpu().numpy().T
    sf.write(str(filepath), data, sample_rate)


if _NEEDS_PATCH:
    torchaudio.load = _load_soundfile
    torchaudio.save = _save_soundfile
