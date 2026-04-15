"""Minimal decoder-only fine-tuning script for DualCodec.

Loads the pretrained DualCodec + w2v-bert-2.0, freezes everything except
the DAC decoder and convnext_decoder, and trains with mel-spectrogram
reconstruction + GAN adversarial loss for a short run.

This is a simplified version of train_mimi.py adapted for DualCodec's
architecture (which requires w2v-bert-2.0 semantic features for encode).

Usage::

    uv run python train/train_dualcodec_decoder.py \\
        --config configs/experiments/dualcodec_hindi_decoder_ft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import eval._audio_compat  # noqa: F401

from train.config_loader import load_config
from train.utils.discriminator import (
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
)

logger = logging.getLogger(__name__)


# ── Dataset ─────────────────────────────────────────────────────────────


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, sample_rate=24000, segment_s=5.0):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_s * sample_rate)
        self.data_dir = Path(manifest_path).parent.parent

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        import eval._audio_compat  # noqa: F401 — patch for workers
        entry = self.manifest[idx]
        path = entry["audio_path"]
        if not Path(path).is_absolute():
            path = str(self.data_dir / path)
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        # Random crop to segment length
        if waveform.shape[-1] > self.segment_samples:
            start = random.randint(0, waveform.shape[-1] - self.segment_samples)
            waveform = waveform[:, start:start + self.segment_samples]
        elif waveform.shape[-1] < self.segment_samples:
            waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
        return waveform.squeeze(0)  # (samples,)


def collate_fn(batch):
    return torch.stack(batch).unsqueeze(1)  # (B, 1, T)


# ── Validation ──────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, val_loader, device):
    """Compute average reconstruction L1 loss on validation set."""
    total_loss, n = 0.0, 0
    for batch in val_loader:
        audio = batch.to(device)
        semantic_codes, acoustic_codes = model.encode(audio)
        recon = model.decode(semantic_codes, acoustic_codes)
        # Match lengths
        min_len = min(audio.shape[-1], recon.shape[-1])
        loss = F.l1_loss(recon[..., :min_len], audio[..., :min_len])
        total_loss += loss.item() * audio.shape[0]
        n += audio.shape[0]
    return total_loss / max(n, 1)


# ── Training ────────────────────────────────────────────────────────────


def train(config: Dict[str, Any]):
    device = torch.device("cuda")

    # Load DualCodec with Inference wrapper (handles w2v-bert-2.0)
    import dualcodec
    logger.info("Loading pretrained DualCodec: %s", config["codec"]["pretrained"])
    raw_model = dualcodec.get_model(config["codec"]["pretrained"])
    model = dualcodec.Inference(raw_model, device=str(device))

    # Freeze encoder-side params (only train decoder)
    frozen, trainable = 0, 0
    for name, param in raw_model.named_parameters():
        if "decoder" in name and "encode" not in name:
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    logger.info(
        "Decoder-only: frozen=%.1fM, trainable=%.1fM",
        frozen / 1e6, trainable / 1e6,
    )

    # Discriminator
    disc = MultiScaleSTFTDiscriminator().to(device)

    # Optimizers (only trainable params)
    lr = float(config["optimizer"]["lr"])
    gen_optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=lr * 0.1)

    # Data
    data_dir = Path(config["dataset"]["local_dir"])
    sr = int(config["codec"]["sample_rate"])
    seg_s = float(config["codec"]["training"].get("segment_s", 5.0))

    train_ds = AudioDataset(data_dir / "train" / "manifest.json", sr, seg_s)
    val_ds = AudioDataset(data_dir / "val" / "manifest.json", sr, seg_s)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=2,
        collate_fn=collate_fn,
    )

    # Training config
    max_steps = config["training"]["max_steps"]
    eval_every = config["training"].get("eval_every", 500)
    save_every = config["training"].get("save_every", 500)
    log_every = config["training"].get("log_every", 50)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    disc_warmup = config.get("discriminator", {}).get("warmup_steps", 200)

    # WandB
    wandb_enabled = config.get("wandb", {}).get("enabled", True)
    if wandb_enabled:
        import wandb
        wandb.init(
            project=config["wandb"].get("project", "codec-finetuning"),
            name=config["wandb"].get("run_name"),
            tags=config["wandb"].get("tags", []),
            config=config,
        )

    # Step 0 validation
    val_loss_0 = validate(model, val_loader, device)
    logger.info("step=0  val_loss=%.4f  (pretrained baseline)", val_loss_0)
    if wandb_enabled:
        wandb.log({"val/loss": val_loss_0}, step=0)

    # Training loop
    raw_model.train()
    data_iter = iter(train_loader)
    step = 0
    start_time = time.monotonic()

    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        audio = batch.to(device)

        # Forward: encode (frozen) → decode (trainable)
        with torch.no_grad():
            semantic_codes, acoustic_codes = model.encode(audio)

        # Decode with gradient
        recon = raw_model.decode_from_codes(semantic_codes, acoustic_codes).float()

        # Match lengths
        min_len = min(audio.shape[-1], recon.shape[-1])
        audio_t = audio[..., :min_len]
        recon_t = recon[..., :min_len]

        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(recon_t, audio_t)

        # Discriminator step (after warmup)
        if step > disc_warmup:
            disc_optimizer.zero_grad()
            d_loss = discriminator_loss(disc, audio_t.detach(), recon_t.detach())
            d_loss.backward()
            disc_optimizer.step()

            # Generator adversarial step
            real_features, fake_features = disc(audio_t), disc(recon_t)
            adv_loss = generator_loss(fake_features)
            fm_loss = feature_matching_loss(real_features, fake_features)
            total_loss = recon_loss + 0.1 * adv_loss + 2.0 * fm_loss
        else:
            total_loss = recon_loss

        gen_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in raw_model.parameters() if p.requires_grad], 1.0
        )
        gen_optimizer.step()

        if step % log_every == 0:
            elapsed = time.monotonic() - start_time
            logger.info(
                "step=%d  loss=%.4f  recon=%.4f  %.1f steps/s",
                step, total_loss.item(), recon_loss.item(), step / elapsed,
            )
            if wandb_enabled:
                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                }, step=step)

        # Validation
        if step % eval_every == 0:
            raw_model.eval()
            val_loss = validate(model, val_loader, device)
            raw_model.train()
            logger.info("step=%d  val_loss=%.4f", step, val_loss)
            if wandb_enabled:
                wandb.log({"val/loss": val_loss}, step=step)

        # Checkpoint
        if step % save_every == 0:
            ckpt_path = output_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                "model_state_dict": {
                    k: v for k, v in raw_model.state_dict().items()
                },
                "step": step,
                "val_loss": val_loss if step % eval_every == 0 else None,
            }, ckpt_path)
            logger.info("Saved: %s", ckpt_path)

    logger.info("Training complete: %d steps", step)
    if wandb_enabled:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
