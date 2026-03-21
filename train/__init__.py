"""Training modules for the codec-finetuning benchmark.

This package provides the core training infrastructure for fine-tuning
neural speech codecs (Mimi, DualCodec, Kanade) on low-resource languages.

Package structure
-----------------
- :mod:`train.config_loader` — YAML config loading with hierarchical
  ``_bases_`` merging, validation, and WandB sweep overrides.
- :mod:`train.optimizer_factory` — Unified factory for all 8 supported
  optimizers (AdamW, RAdam, Lion, Prodigy, Schedule-Free AdamW, SOAP,
  Adan, Muon) with metadata flags for the training loop.
- :mod:`train.train_mimi` — Main training script for the Mimi codec
  with adversarial training, EMA, gradient-balanced losses, and
  augmentation.
- :mod:`train.sweep_agent` — WandB sweep agent entry point that
  dispatches to the correct codec training pipeline.
- :mod:`train.utils` — Shared utilities: augmentation, discriminator,
  EMA, loss balancer, and the vendored Muon optimizer.

License: MIT
"""
