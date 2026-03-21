"""Training utilities for codec-finetuning.

This sub-package bundles reusable components shared across codec
training pipelines:

- :mod:`~train.utils.augmentation` — waveform-level data augmentation
  with preset configurations (speed perturbation, pitch shift, noise
  injection, gain jitter, chunk reversal).
- :mod:`~train.utils.discriminator` — multi-scale STFT discriminator
  and associated loss functions (hinge, feature-matching, R1 penalty).
- :mod:`~train.utils.ema` — exponential moving average model tracker
  with swap-in/restore for evaluation.
- :mod:`~train.utils.loss_balancer` — AudioCraft-style gradient-normalised
  multi-loss balancer.
- :mod:`~train.utils.muon` — vendored Muon optimizer (Newton-Schulz
  orthogonalised SGD momentum).

Re-exports the key classes and functions so that downstream code can do::

    from train.utils import Muon, EMAModel, LossBalancer
    from train.utils import MultiScaleSTFTDiscriminator

License: MIT
"""

from train.utils.augmentation import AugmentationConfig, augment_waveform, resolve_preset
from train.utils.discriminator import (
    MultiScaleSTFTDiscriminator,
    STFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
    r1_penalty,
)
from train.utils.ema import EMAModel
from train.utils.loss_balancer import LossBalancer
from train.utils.muon import Muon

__all__ = [
    # augmentation
    "AugmentationConfig",
    "augment_waveform",
    "resolve_preset",
    # discriminator
    "MultiScaleSTFTDiscriminator",
    "STFTDiscriminator",
    "discriminator_loss",
    "feature_matching_loss",
    "generator_loss",
    "r1_penalty",
    # ema
    "EMAModel",
    # loss_balancer
    "LossBalancer",
    # muon
    "Muon",
]
