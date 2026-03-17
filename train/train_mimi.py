"""Main training script for fine-tuning the Mimi neural speech codec.

Implements the full training loop with adversarial training, EMA model
tracking, gradient-balanced multi-loss optimisation, and waveform
augmentation.  Supports all 8 optimizers from the factory, including
Schedule-Free variants that require special eval-mode handling.

Usage::

    uv run python train/train_mimi.py \
        --config configs/experiments/mimi_turkish_sample.yaml

    # Resume from a checkpoint:
    uv run python train/train_mimi.py \
        --config configs/experiments/mimi_turkish_sample.yaml \
        --resume outputs/mimi_turkish_sample/checkpoint_step_2000.pt

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
)
from torch.utils.data import DataLoader, Dataset

from train.config_loader import load_config
from train.optimizer_factory import create_optimizer
from train.utils.augmentation import (
    AugmentationConfig,
    augment_waveform,
    resolve_preset,
)
from train.utils.discriminator import (
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
    r1_penalty,
)
from train.utils.ema import EMAModel
from train.utils.loss_balancer import LossBalancer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Configures Python's ``random``, NumPy, and PyTorch (CPU + CUDA) to
    use the same seed, and enables deterministic cuDNN behaviour.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


class _AudioManifestDataset(Dataset):
    """Reads audio paths from a JSON manifest and loads waveforms.

    Each manifest entry is expected to have at least an ``"audio_path"``
    key pointing to a ``.wav`` or ``.flac`` file on disk.

    Args:
        manifest_path: Path to the JSON manifest file.
        base_dir: Root directory that ``audio_path`` entries are relative to.
        segment_samples: Number of samples to crop/pad each utterance to.
        sample_rate: Target sample rate for resampling.
    """

    def __init__(
        self,
        manifest_path: str,
        base_dir: str,
        segment_samples: int,
        sample_rate: int,
    ) -> None:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            self.entries: List[Dict[str, Any]] = json.load(fh)
        self.base_dir = Path(base_dir)
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        """Return the number of utterances in the manifest."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and pre-process a single audio utterance.

        Args:
            idx: Index into the manifest.

        Returns:
            A dict with key ``"audio"`` containing a tensor of shape
            ``(1, segment_samples)``.
        """
        import soundfile as sf

        entry = self.entries[idx]
        audio_path = self.base_dir / entry["audio_path"]
        audio_np, sr = sf.read(str(audio_path), dtype="float32")

        # Force mono.
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)

        # Resample if needed.
        if sr != self.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # Crop or pad to fixed length.
        if waveform.shape[-1] > self.segment_samples:
            start = random.randint(
                0, waveform.shape[-1] - self.segment_samples
            )
            waveform = waveform[:, start : start + self.segment_samples]
        elif waveform.shape[-1] < self.segment_samples:
            pad = self.segment_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad))

        return {"audio": waveform}


def _load_dataset(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Load train and validation datasets from manifest JSON files.

    Expects ``config["dataset"]["local_dir"]`` to contain split
    subdirectories (``train/``, ``val/``) each with a ``manifest.json``.

    Args:
        config: Full experiment config dict.

    Returns:
        A ``(train_loader, val_loader)`` tuple of DataLoaders.

    Raises:
        FileNotFoundError: If manifest files are missing.
    """
    local_dir = Path(config["dataset"]["local_dir"])
    sr: int = config["codec"]["sample_rate"]
    segment_s: float = config["codec"]["training"]["segment_s"]
    segment_samples = int(sr * segment_s)
    batch_size: int = config["codec"]["training"]["micro_batch_size"]

    train_manifest = local_dir / "train" / "manifest.json"
    val_manifest = local_dir / "val" / "manifest.json"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train manifest not found: {train_manifest}"
        )
    if not val_manifest.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: {val_manifest}"
        )

    train_ds = _AudioManifestDataset(
        str(train_manifest), str(local_dir), segment_samples, sr
    )
    val_ds = _AudioManifestDataset(
        str(val_manifest), str(local_dir), segment_samples, sr
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[LRScheduler]:
    """Create a learning-rate scheduler from the config.

    Supports ``"cosine"``, ``"linear"``, and ``"constant"`` schedules,
    each with an optional linear warmup phase.  Returns ``None`` when the
    optimizer metadata indicates that external scheduling should be
    disabled (e.g. Prodigy, Schedule-Free AdamW).

    Args:
        optimizer: The generator optimizer.
        config: Full experiment config dict.
        metadata: Metadata dict returned by
            :func:`train.optimizer_factory.create_optimizer`.

    Returns:
        An ``LRScheduler`` instance, or ``None`` if scheduling is disabled.
    """
    if metadata.get("disable_scheduler"):
        logger.info("Scheduler disabled by optimizer metadata.")
        return None

    sched_cfg = config.get("scheduler", {})
    name: str = sched_cfg.get("name", "cosine")
    warmup_steps: int = int(sched_cfg.get("warmup_steps", 500))
    max_steps: int = int(config["training"]["max_steps"])
    min_lr_ratio: float = float(sched_cfg.get("min_lr_ratio", 0.01))

    if name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(max_steps - warmup_steps, 1),
            eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
        )
    elif name == "linear":

        def _linear_decay(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(
                max_steps - warmup_steps, 1
            )
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

        scheduler = LambdaLR(optimizer, lr_lambda=_linear_decay)
    elif name == "constant":

        def _constant_with_warmup(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=_constant_with_warmup)
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            "Supported: 'cosine', 'linear', 'constant'."
        )

    # Wrap with warmup for cosine (Lambda variants handle it inline).
    if name == "cosine" and warmup_steps > 0:

        def _warmup_then_cosine(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0  # CosineAnnealingLR handles the decay.

        scheduler = LambdaLR(optimizer, lr_lambda=_warmup_then_cosine)

    return scheduler


# ---------------------------------------------------------------------------
# Augmentation batch helper
# ---------------------------------------------------------------------------


def augment_batch(
    audio: torch.Tensor,
    sr: int,
    aug_config: AugmentationConfig,
) -> torch.Tensor:
    """Apply waveform augmentation to each sample in a batch.

    Args:
        audio: Batch of waveforms with shape ``(batch, 1, samples)``.
        sr: Sample rate in Hz.
        aug_config: Resolved augmentation config.

    Returns:
        Augmented batch with the same shape.
    """
    augmented = []
    for i in range(audio.shape[0]):
        wav = augment_waveform(audio[i], sr, aug_config)
        augmented.append(wav)
    return torch.stack(augmented, dim=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: Dict[str, Any],
) -> float:
    """Run a full validation pass and return the average loss.

    Computes only the reconstruction (L1) loss across the validation
    set; adversarial and feature-matching losses are excluded since the
    discriminator is not evaluated.

    Args:
        model: The Mimi codec model.
        val_loader: Validation DataLoader.
        config: Full experiment config dict.

    Returns:
        Average validation reconstruction loss as a float.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in val_loader:
            audio = batch["audio"].cuda(non_blocking=True)
            output = model(audio)
            reconstructed = output.audio_values
            loss = F.l1_loss(reconstructed, audio)
            total_loss += loss.item() * audio.shape[0]
            count += audio.shape[0]

    avg_loss = total_loss / max(count, 1)
    model.train()
    return avg_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMAModel],
    step: int,
    config: Dict[str, Any],
    disc: Optional[torch.nn.Module] = None,
    disc_optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Save a training checkpoint atomically (write-then-rename).

    The checkpoint is first written to a temporary file in the same
    directory, then renamed to the final path to avoid partial writes
    on crashes.

    Args:
        model: The Mimi codec model.
        optimizer: Generator optimizer.
        ema: EMA model tracker, or ``None``.
        step: Current training step.
        config: Full experiment config dict.
        disc: Discriminator module, or ``None``.
        disc_optimizer: Discriminator optimizer, or ``None``.
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"checkpoint_step_{step}.pt"
    payload: Dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    if disc is not None:
        payload["disc_state_dict"] = disc.state_dict()
    if disc_optimizer is not None:
        payload["disc_optimizer_state_dict"] = disc_optimizer.state_dict()

    # Atomic write: write to tmp file, then rename.
    fd, tmp_path = tempfile.mkstemp(
        dir=str(output_dir), suffix=".pt.tmp"
    )
    try:
        os.close(fd)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, str(ckpt_path))
        logger.info("Saved checkpoint: %s", ckpt_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class _EarlyStopping:
    """Tracks validation metric and signals when to stop.

    Args:
        patience: Number of evaluations without improvement before
            stopping.
        mode: ``"min"`` if lower is better, ``"max"`` if higher is better.
    """

    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter: int = 0

    def __call__(self, metric: float) -> bool:
        """Check whether training should stop.

        Args:
            metric: Current validation metric value.

        Returns:
            ``True`` if patience is exhausted, ``False`` otherwise.
        """
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric < self.best
            if self.mode == "min"
            else metric > self.best
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config: Dict[str, Any]) -> None:
    """Run the full Mimi fine-tuning training loop.

    Orchestrates model loading, optimizer/scheduler creation, adversarial
    training with a multi-scale STFT discriminator, EMA tracking,
    gradient-balanced loss combination, augmentation, validation with
    early stopping, and atomic checkpointing.

    Args:
        config: Fully merged and validated experiment config dict, as
            returned by :func:`train.config_loader.load_config`.

    Raises:
        RuntimeError: On unrecoverable CUDA errors or repeated NaN losses.
    """
    # ── Setup ────────────────────────────────────────────────────────────
    seed: int = config["training"]["seed"]
    _set_seed(seed)
    torch.set_float32_matmul_precision("high")

    # WandB
    wandb_enabled: bool = config.get("wandb", {}).get("enabled", False)
    if wandb_enabled:
        import wandb

        wandb.init(
            project=config["wandb"].get("project", "codec-finetuning"),
            config=config,
            tags=config["wandb"].get("tags", []),
            name=config["wandb"].get("run_name"),
        )

    # Model
    from transformers import MimiModel

    logger.info(
        "Loading pretrained Mimi: %s", config["codec"]["pretrained"]
    )
    model = MimiModel.from_pretrained(config["codec"]["pretrained"])
    model = model.cuda()

    if config["training"].get("compile", False):
        compile_mode = config["training"].get(
            "compile_mode", "default"
        )
        logger.info("Compiling model with mode=%s", compile_mode)
        model = torch.compile(model, mode=compile_mode)

    # Discriminator
    discriminator = MultiScaleSTFTDiscriminator().cuda()

    # Optimizers
    optimizer, metadata = create_optimizer(model, config)
    gen_lr: float = float(config["optimizer"].get("lr", 1e-4))
    disc_lr_ratio: float = float(
        config.get("discriminator", {}).get("lr_ratio", 0.5)
    )
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=gen_lr * disc_lr_ratio
    )

    # Scheduler
    scheduler = _create_scheduler(optimizer, config, metadata)

    # EMA
    ema: Optional[EMAModel] = None
    ema_cfg = config.get("ema", {})
    if ema_cfg.get("enabled", False):
        ema = EMAModel(
            model,
            decay=float(ema_cfg.get("decay", 0.999)),
            start_step=int(ema_cfg.get("start_step", 1000)),
        )

    # Loss balancer
    loss_balancer: Optional[LossBalancer] = None
    lb_cfg = config.get("loss_balancer", {})
    if lb_cfg.get("enabled", False):
        loss_weights: Dict[str, float] = {
            "reconstruction": float(
                config["codec"]["losses"]["reconstruction_weight"]
            ),
            "commitment": float(
                config["codec"]["losses"]["commit_loss_weight"]
            ),
            "adversarial": float(
                config["codec"]["losses"]["adversarial_weight"]
            ),
            "feature_matching": float(
                config["codec"]["losses"]["feature_matching_weight"]
            ),
        }
        loss_balancer = LossBalancer(
            weights=loss_weights,
            ema_decay=float(lb_cfg.get("ema_decay", 0.999)),
        )

    # Augmentation
    aug_cfg = config.get("augmentation", {})
    aug_config: Optional[AugmentationConfig] = None
    aug_preset: str = aug_cfg.get("preset", "none")
    if aug_preset != "none":
        aug_config = resolve_preset(AugmentationConfig(preset=aug_preset))

    sr: int = config["codec"]["sample_rate"]

    # Dataset
    train_loader, val_loader = _load_dataset(config)

    # Training params
    max_steps: int = config["training"]["max_steps"]
    log_every: int = config["training"].get("log_every", 50)
    eval_every: int = config["training"].get("eval_every", 500)
    save_every: int = config["training"].get("save_every", 1000)
    grad_accum_steps: int = config["training"].get("grad_accum_steps", 4)
    max_grad_norm: float = float(
        config["training"].get("max_grad_norm", 1.0)
    )
    disc_warmup: int = int(
        config.get("discriminator", {}).get("warmup_steps", 500)
    )
    r1_weight: float = float(
        config.get("discriminator", {}).get("r1_penalty", 10.0)
    )
    commit_weight: float = float(
        config["codec"]["losses"].get("commit_loss_weight", 0.0)
    )
    recon_weight: float = float(
        config["codec"]["losses"].get("reconstruction_weight", 1.0)
    )
    adv_weight: float = float(
        config["codec"]["losses"].get("adversarial_weight", 0.1)
    )
    fm_weight: float = float(
        config["codec"]["losses"].get("feature_matching_weight", 2.0)
    )

    # Early stopping
    es_cfg = config.get("early_stopping", {})
    early_stopping: Optional[_EarlyStopping] = None
    if es_cfg.get("enabled", False):
        early_stopping = _EarlyStopping(
            patience=int(es_cfg.get("patience", 5)),
            mode=es_cfg.get("mode", "min"),
        )

    # Resume from checkpoint
    start_step = 0
    # (resume path is passed via config["_resume_path"] set by CLI)
    resume_path = config.get("_resume_path")
    if resume_path and Path(resume_path).exists():
        logger.info("Resuming from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location="cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if "disc_state_dict" in ckpt:
            discriminator.load_state_dict(ckpt["disc_state_dict"])
        if "disc_optimizer_state_dict" in ckpt:
            disc_optimizer.load_state_dict(
                ckpt["disc_optimizer_state_dict"]
            )
        logger.info("Resumed at step %d", start_step)

    # ── Signal handlers for graceful shutdown ────────────────────────────
    _shutdown_requested = False

    def _signal_handler(signum: int, frame: Any) -> None:
        nonlocal _shutdown_requested
        logger.warning(
            "Received signal %d, saving checkpoint before exit...", signum
        )
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── Training loop ────────────────────────────────────────────────────
    data_iter = iter(train_loader)
    consecutive_nans = 0
    last_ckpt_path = resume_path
    model.train()
    optimizer.zero_grad()

    try:
        for step in range(start_step + 1, max_steps + 1):
            if _shutdown_requested:
                logger.info("Shutdown requested — saving and exiting.")
                save_checkpoint(
                    model, optimizer, ema, step, config,
                    disc=discriminator, disc_optimizer=disc_optimizer,
                )
                break

            # Get batch, handle epoch rollover.
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            audio = batch["audio"].cuda(non_blocking=True)

            # Augment
            if aug_config is not None:
                audio = augment_batch(audio, sr, aug_config)

            # ── Generator forward ────────────────────────────────────
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(audio)
                reconstructed = output.audio_values

                losses: Dict[str, torch.Tensor] = {}
                losses["reconstruction"] = (
                    F.l1_loss(reconstructed, audio) * recon_weight
                )

                if commit_weight > 0 and hasattr(output, "quantizer_loss") and output.quantizer_loss is not None:
                    losses["commitment"] = (
                        output.quantizer_loss * commit_weight
                    )

            # ── Discriminator step (after warmup) ────────────────────
            if step > disc_warmup:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    real_logits, real_features = discriminator(audio)
                    fake_logits, fake_features = discriminator(
                        reconstructed.detach()
                    )
                    disc_loss = discriminator_loss(
                        real_logits, fake_logits
                    )

                    # R1 penalty every 16 steps.
                    if r1_weight > 0 and step % 16 == 0:
                        audio_r1 = audio.detach().requires_grad_(True)
                        r1 = r1_penalty(audio_r1, discriminator)
                        disc_loss = disc_loss + r1_weight * r1

                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                # Generator adversarial + feature matching.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake_logits_gen, fake_features_gen = discriminator(
                        reconstructed
                    )
                    losses["adversarial"] = (
                        generator_loss(fake_logits_gen) * adv_weight
                    )
                    losses["feature_matching"] = (
                        feature_matching_loss(
                            real_features, fake_features_gen
                        )
                        * fm_weight
                    )

            # ── Balance and combine losses ───────────────────────────
            if loss_balancer is not None:
                total_loss = loss_balancer.balance(
                    losses, reconstructed
                )
            else:
                total_loss = sum(losses.values())

            # NaN detection
            if torch.isnan(total_loss):
                consecutive_nans += 1
                logger.warning(
                    "NaN loss at step %d (%d consecutive)",
                    step, consecutive_nans,
                )
                if consecutive_nans >= 3:
                    logger.error(
                        "3 consecutive NaN losses — halving LR and "
                        "reloading last checkpoint."
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.5
                    if (
                        last_ckpt_path
                        and Path(last_ckpt_path).exists()
                    ):
                        ckpt = torch.load(
                            last_ckpt_path, map_location="cuda"
                        )
                        model.load_state_dict(
                            ckpt["model_state_dict"]
                        )
                    consecutive_nans = 0
                optimizer.zero_grad()
                continue
            else:
                consecutive_nans = 0

            # ── Backward + gradient accumulation ─────────────────────
            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            # EMA update
            if ema is not None:
                ema.update(model, step)

            # ── Logging ──────────────────────────────────────────────
            if step % log_every == 0:
                log_dict: Dict[str, Any] = {
                    f"train/{k}": v.item()
                    for k, v in losses.items()
                }
                log_dict["train/total_loss"] = (
                    total_loss.item()
                )
                log_dict["train/lr"] = (
                    optimizer.param_groups[0]["lr"]
                )
                log_dict["train/step"] = step
                if wandb_enabled:
                    import wandb

                    wandb.log(log_dict, step=step)
                else:
                    logger.info(
                        "step=%d  loss=%.4f  lr=%.2e",
                        step,
                        total_loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )

            # ── Validation ───────────────────────────────────────────
            if step % eval_every == 0:
                # Schedule-Free: switch to eval mode before validation.
                if metadata.get("needs_eval_mode"):
                    optimizer.eval()

                val_loss = validate(model, val_loader, config)
                logger.info(
                    "step=%d  val_loss=%.4f", step, val_loss
                )

                if metadata.get("needs_eval_mode"):
                    optimizer.train()

                if wandb_enabled:
                    import wandb

                    wandb.log({"val/loss": val_loss}, step=step)

                if early_stopping is not None and early_stopping(
                    val_loss
                ):
                    logger.info(
                        "Early stopping at step %d", step
                    )
                    save_checkpoint(
                        model, optimizer, ema, step, config,
                        disc=discriminator,
                        disc_optimizer=disc_optimizer,
                    )
                    break

            # ── Checkpoint ───────────────────────────────────────────
            if step % save_every == 0:
                save_checkpoint(
                    model, optimizer, ema, step, config,
                    disc=discriminator,
                    disc_optimizer=disc_optimizer,
                )
                last_ckpt_path = str(
                    Path(config["output_dir"])
                    / f"checkpoint_step_{step}.pt"
                )

    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            logger.warning(
                "CUDA OOM at step %d. Saving emergency checkpoint.",
                step,
            )
            torch.cuda.empty_cache()
            save_checkpoint(
                model, optimizer, ema, step, config,
                disc=discriminator, disc_optimizer=disc_optimizer,
            )
            raise
        else:
            logger.error("RuntimeError at step %d: %s", step, exc)
            save_checkpoint(
                model, optimizer, ema, step, config,
                disc=discriminator, disc_optimizer=disc_optimizer,
            )
            raise
    except Exception as exc:
        logger.error(
            "Unexpected error at step %d: %s", step, exc
        )
        try:
            save_checkpoint(
                model, optimizer, ema, step, config,
                disc=discriminator, disc_optimizer=disc_optimizer,
            )
        except Exception:
            logger.error("Failed to save emergency checkpoint.")
        raise

    # ── Cleanup ──────────────────────────────────────────────────────────
    if wandb_enabled:
        import wandb

        wandb.finish()

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune the Mimi neural speech codec.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    if args.resume:
        config["_resume_path"] = args.resume

    train(config)


if __name__ == "__main__":
    main()
