"""Training utilities for codec-finetuning.

Re-exports the key classes and functions from submodules so that
downstream code can do::

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
