"""Multi-scale STFT discriminator for adversarial codec training.

Implements a stack of STFT-based discriminators operating at different
time-frequency resolutions.  Each sub-discriminator converts the waveform
to a complex spectrogram, stacks real and imaginary parts as a 2-channel
image, and passes it through a lightweight ConvNet that outputs per-patch
logits together with intermediate feature maps (for feature-matching loss).

Also provides the four loss helpers commonly used with this architecture:
hinge-based discriminator / generator losses, L1 feature-matching loss,
and R1 gradient penalty.

Usage::

    from train.utils.discriminator import (
        MultiScaleSTFTDiscriminator,
        discriminator_loss,
        generator_loss,
        feature_matching_loss,
    )

    disc = MultiScaleSTFTDiscriminator()
    real_logits, real_feats = disc(real_audio)
    fake_logits, fake_feats = disc(fake_audio.detach())

    d_loss = discriminator_loss(real_logits, fake_logits)
    g_loss = generator_loss(fake_logits)
    fm_loss = feature_matching_loss(real_feats, fake_feats)

Dependencies:
    torch

License: MIT
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Single-scale discriminator
# ---------------------------------------------------------------------------


class STFTDiscriminator(nn.Module):
    """Single-scale STFT-based patch discriminator.

    Converts a mono waveform into a complex spectrogram, stacks real and
    imaginary components as two channels, and passes the result through four
    ``Conv2d`` layers with ``LeakyReLU`` activation followed by a final
    1-channel projection.

    Args:
        n_fft: FFT size.
        hop_length: STFT hop length.
        win_length: STFT window length.

    Example::

        >>> disc = STFTDiscriminator(1024, 256, 1024)
        >>> logits, feats = disc(torch.randn(2, 1, 24000))
        >>> logits.shape  # (batch, 1, freq_bins, time_frames)
        torch.Size([...])
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Register a Hann window as a buffer so it moves with the model.
        self.register_buffer("window", torch.hann_window(win_length))

        # 2-channel input (real + imag) → increasing channel widths.
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        ])
        self.out_conv = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Run the discriminator on a waveform.

        Args:
            x: Audio tensor of shape ``(batch, 1, samples)`` or
                ``(batch, samples)``.

        Returns:
            A tuple ``(logits, feature_maps)`` where *logits* has shape
            ``(batch, 1, F', T')`` and *feature_maps* is a list of
            intermediate activations (one per conv layer), useful for
            feature-matching loss.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Flatten to mono for STFT.
        wav = x.squeeze(1)  # (B, T)

        # Compute complex STFT; output shape (B, freq_bins, time_frames).
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        # Stack real and imaginary as 2-channel image: (B, 2, F, T).
        x = torch.stack([spec.real, spec.imag], dim=1)

        feature_maps: List[Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=0.2)
            feature_maps.append(x)

        logits = self.out_conv(x)
        return logits, feature_maps


# ---------------------------------------------------------------------------
# Multi-scale wrapper
# ---------------------------------------------------------------------------


class MultiScaleSTFTDiscriminator(nn.Module):
    """Wraps several :class:`STFTDiscriminator` at different STFT scales.

    Using multiple resolutions forces the generator to produce plausible
    outputs across both coarse (large FFT) and fine (small FFT) spectral
    views.

    Args:
        scales: List of ``(n_fft, hop_length, win_length)`` tuples.
            Defaults to three standard scales if ``None``.

    Example::

        >>> disc = MultiScaleSTFTDiscriminator()
        >>> logits, feats = disc(torch.randn(4, 1, 24000))
        >>> len(logits)  # one set of logits per scale
        3
    """

    def __init__(
        self,
        scales: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        super().__init__()
        if scales is None:
            # Coarse → fine: large FFT captures low-freq structure,
            # small FFT captures high-freq transients.
            scales = [
                (2048, 512, 2048),
                (1024, 256, 1024),
                (512, 128, 512),
            ]
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft, hop, win) for n_fft, hop, win in scales
        ])

    def forward(
        self, x: Tensor
    ) -> Tuple[List[Tensor], List[List[Tensor]]]:
        """Run all sub-discriminators.

        Args:
            x: Audio tensor of shape ``(batch, 1, samples)``.

        Returns:
            A tuple ``(all_logits, all_feature_maps)`` where each list
            has one entry per scale.
        """
        all_logits: List[Tensor] = []
        all_feature_maps: List[List[Tensor]] = []
        for disc in self.discriminators:
            logits, feats = disc(x)
            all_logits.append(logits)
            all_feature_maps.append(feats)
        return all_logits, all_feature_maps


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def discriminator_loss(
    real_logits: List[Tensor],
    fake_logits: List[Tensor],
) -> Tensor:
    """Hinge loss for the discriminator.

    Encourages the discriminator to push real logits above +1 and fake
    logits below -1.

    Args:
        real_logits: Per-scale discriminator outputs on real audio.
        fake_logits: Per-scale discriminator outputs on generated audio.

    Returns:
        Scalar discriminator loss.

    Example::

        >>> d_loss = discriminator_loss(real_logits, fake_logits)
    """
    loss = torch.tensor(0.0, device=real_logits[0].device)
    for rl, fl in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - rl).mean() + F.relu(1.0 + fl).mean()
    return loss


def generator_loss(fake_logits: List[Tensor]) -> Tensor:
    """Hinge loss for the generator.

    The generator wants the discriminator to classify its outputs as real,
    i.e. push fake logits above 0.

    Args:
        fake_logits: Per-scale discriminator outputs on generated audio.

    Returns:
        Scalar generator loss.

    Example::

        >>> g_loss = generator_loss(fake_logits)
    """
    loss = torch.tensor(0.0, device=fake_logits[0].device)
    for fl in fake_logits:
        loss = loss - fl.mean()  # Maximise fake logits.
    return loss


def feature_matching_loss(
    real_features: List[List[Tensor]],
    fake_features: List[List[Tensor]],
) -> Tensor:
    """L1 feature-matching loss across all scales and layers.

    Minimises the L1 distance between intermediate discriminator
    activations on real vs. generated audio, encouraging perceptually
    similar outputs.

    Args:
        real_features: Nested list ``[scale][layer]`` of feature maps
            from real audio.
        fake_features: Corresponding feature maps from generated audio.

    Returns:
        Scalar feature-matching loss.

    Example::

        >>> fm = feature_matching_loss(real_feats, fake_feats)
    """
    loss = torch.tensor(0.0, device=real_features[0][0].device)
    count = 0
    for rf_scale, ff_scale in zip(real_features, fake_features):
        for rf, ff in zip(rf_scale, ff_scale):
            loss = loss + F.l1_loss(ff, rf.detach())
            count += 1
    # Average over all (scale, layer) pairs.
    return loss / max(count, 1)


def r1_penalty(
    real_audio: Tensor,
    discriminator: MultiScaleSTFTDiscriminator,
) -> Tensor:
    """R1 gradient penalty on real data.

    Penalises the squared gradient norm of the discriminator output
    with respect to the real input, acting as a zero-centred gradient
    penalty that stabilises GAN training.

    Args:
        real_audio: Real audio tensor with ``requires_grad=True``.
            Shape ``(batch, 1, samples)``.
        discriminator: The multi-scale discriminator.

    Returns:
        Scalar R1 penalty.

    Raises:
        RuntimeError: If *real_audio* does not require gradients.

    Example::

        >>> audio = torch.randn(2, 1, 24000, requires_grad=True)
        >>> penalty = r1_penalty(audio, disc)
    """
    if not real_audio.requires_grad:
        raise RuntimeError("real_audio must have requires_grad=True for R1 penalty")

    all_logits, _ = discriminator(real_audio)

    # Sum logits across all scales before computing gradients.
    logit_sum = sum(lg.sum() for lg in all_logits)
    (grad,) = torch.autograd.grad(
        outputs=logit_sum,
        inputs=real_audio,
        create_graph=True,
    )

    # Squared L2 norm of the gradient, averaged over the batch.
    penalty = grad.square().sum(dim=[1, 2]).mean() if grad.dim() == 3 else grad.square().mean()
    return penalty
