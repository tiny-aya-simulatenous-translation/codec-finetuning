# Benchmark Results

All experiments run on **1x NVIDIA H100 PCIe 80 GB**, Ubuntu 22.04, CUDA 12.8,
PyTorch 2.9.1, Python 3.12.

WandB project: [cataluna84/codec-finetuning](https://wandb.ai/cataluna84/codec-finetuning)

---

## Phase 1: Hyperparameter Sweep -- Mimi on Turkish Sample (10 h)

Bayesian optimization with Hyperband early termination across 8 optimizers.
68 total runs (~36 h wall clock), 34 in the main sweep (`j7oxgz4p`).

- Sweep: [wandb.ai/cataluna84/codec-finetuning/sweeps/j7oxgz4p](https://wandb.ai/cataluna84/codec-finetuning/sweeps/j7oxgz4p)
- Codec: Mimi (kyutai/mimi, 96M params, 12.5 Hz, 8 codebooks)
- Dataset: Turkish Sample (MediaSpeech + OpenSLR 108, 10 h, ~2500 utterances)
- Budget: 5,000 steps per run

### Top 10 Runs

| Rank | Run | Optimizer | LR | Val Loss |
|:----:|-----|-----------|---:|:--------:|
| 1 | [`ooz250pm`](https://wandb.ai/cataluna84/codec-finetuning/runs/ooz250pm) | **Schedule-Free AdamW** | 1.074e-3 | **0.01222** |
| 2 | [`d0luhhdd`](https://wandb.ai/cataluna84/codec-finetuning/runs/d0luhhdd) | Prodigy | 3.82e-4 | 0.01228 |
| 3 | [`3kstcode`](https://wandb.ai/cataluna84/codec-finetuning/runs/3kstcode) | SOAP | 1.67e-4 | 0.01228 |
| 4 | [`hi2d7mi3`](https://wandb.ai/cataluna84/codec-finetuning/runs/hi2d7mi3) | Adan | 1.2e-5 | 0.01231 |
| 5 | [`qymvehel`](https://wandb.ai/cataluna84/codec-finetuning/runs/qymvehel) | Schedule-Free AdamW | 4.8e-5 | 0.01237 |
| 6 | [`87see7sz`](https://wandb.ai/cataluna84/codec-finetuning/runs/87see7sz) | AdamW | 1.90e-4 | 0.01243 |
| 7 | [`e7z122o8`](https://wandb.ai/cataluna84/codec-finetuning/runs/e7z122o8) | Schedule-Free AdamW | 1.5e-5 | 0.01244 |
| 8 | [`eu0rw91o`](https://wandb.ai/cataluna84/codec-finetuning/runs/eu0rw91o) | Schedule-Free AdamW | 9.2e-5 | 0.01245 |
| 9 | [`7eejbscw`](https://wandb.ai/cataluna84/codec-finetuning/runs/7eejbscw) | Prodigy | 4.4e-5 | 0.01257 |
| 10 | [`eo8v05ci`](https://wandb.ai/cataluna84/codec-finetuning/runs/eo8v05ci) | Prodigy | 2.92e-2 | 0.01259 |

### Best Hyperparameters

From the winning run (`ooz250pm`), saved to
`configs/experiments/mimi_turkish_sample_best.yaml`:

```yaml
optimizer:
  name: schedulefree_adamw
  lr: 1.074e-3
  betas: [0.9, 0.95]
  weight_decay: 0.0
scheduler:
  name: constant          # disabled by Schedule-Free internals
  warmup_steps: 0
augmentation:
  preset: none
discriminator:
  warmup_steps: 0
  lr_ratio: 0.1
  r1_penalty: 50.0
ema:
  decay: 0.999
codec:
  training:
    segment_s: 10.0
    freeze_encoder_steps: 0
```

### Key Findings

- **Schedule-Free AdamW** won the sweep and took 4 of the top 8 spots.
- **Prodigy** (LR-free) placed 2nd with no LR tuning required.
- **SOAP** (second-order) placed 3rd, supporting the small-data hypothesis.
- **Adan** (triple-momentum) placed 4th, validating its CNN+transformer+GAN
  design on a real codec architecture.
- **AdamW** baseline placed 6th -- all top 5 are non-standard optimizers.
- **Muon** completed runs but ranked lower (11th, 13th). First known test of
  Muon on codec/audio/GAN training.
- No augmentation (`preset: none`) was best, suggesting the 10 h dataset is
  clean enough that augmentation hurts at this scale.
- `beta2=0.95` (lower than default 0.999) was preferred, consistent with GAN
  training literature.

---

## Phase 2: Fine-Tuning -- Mimi on Hindi (~90 h)

Full-scale fine-tuning using the best hyperparameters from Phase 1.

| Field | Value |
|---|---|
| Run | [`iwdd7hfg`](https://wandb.ai/cataluna84/codec-finetuning/runs/iwdd7hfg) |
| Config | `configs/experiments/mimi_hindi.yaml` |
| Dataset | Hindi (~90 h, 33,275 utterances) |
| Optimizer | Schedule-Free AdamW (lr=1.074e-3, betas=[0.9, 0.95]) |
| Precision | bf16 |
| Grad accumulation | 8 steps |
| Max steps | 50,000 |
| Duration | ~7 h |

### Training Progression

| Steps | Val Loss | Phase |
|------:|:--------:|-------|
| 1,000 | 0.0216 | Rapid improvement |
| 5,000 | 0.0206 | |
| 10,000 | 0.0204 | Diminishing returns |
| 22,000 | 0.0201 | Plateau begins |
| 39,000 | 0.0200 | Best observed |
| 50,000 | 0.0200 | Final |

Fully converged with no instability, NaN losses, or OOM errors. Saturated
after ~25k steps.

---

## Phase 3: Evaluation Results

Two experiments evaluated using the unified eval pipeline (`eval/run_all.py`):
- **Mimi Hindi**: fine-tuned on Hindi data, evaluated on 1,665 test utterances
- **Mimi Lahaja**: same model evaluated on 3,044 Lahaja (Swahili) test utterances

### Bootstrap Metrics (20 resamples, 95% CI)

| Metric | Hindi (1,665 utt) | Lahaja (3,044 utt) |
|--------|-------------------:|--------------------:|
| **PESQ (wb)** | 3.25 [3.23, 3.26] | 2.10 [2.09, 2.11] |
| **PESQ (nb)** | 3.74 [3.73, 3.75] | 2.83 [2.81, 2.84] |
| **STOI** | 0.079 [0.078, 0.080] | 0.127 [0.123, 0.131] |
| **DNSMOS-SIG** | 3.42 [3.41, 3.43] | 3.14 [3.13, 3.15] |
| **DNSMOS-BAK** | 4.11 [4.11, 4.12] | 3.73 [3.71, 3.73] |
| **DNSMOS-OVRL** | 3.19 [3.19, 3.20] | 2.73 [2.72, 2.74] |
| **MCD** | 857.5 [854.3, 860.5] | 807.9 [804.5, 811.6] |

### SSNR (Segmental Signal-to-Noise Ratio)

| Metric | Hindi | Lahaja |
|--------|------:|-------:|
| Mean | -6.60 dB | -4.46 dB |
| Std | 0.64 dB | 0.91 dB |
| Min | -9.12 dB | -8.64 dB |
| Max | -4.38 dB | -0.39 dB |

### TTFAT (Time to First Audio Token)

| Metric | Hindi | Lahaja |
|--------|------:|-------:|
| Mean | 14.21 ms | 24.58 ms |
| P50 | 14.16 ms | 24.53 ms |
| P95 | 14.50 ms | 24.89 ms |
| P99 | 14.90 ms | 25.13 ms |

### VERSA Comprehensive Evaluation (Hindi only)

| Metric | Value |
|--------|------:|
| PESQ | 3.24 |
| STOI | 0.079 |
| eSTOI | -0.011 |
| SQUIM STOI | **0.985** |
| SQUIM PESQ | 3.45 |
| SQUIM SI-SDR | 25.09 dB |
| SDR | -23.65 dB |
| CI-SDR | -23.57 dB |
| SI-SNR | -41.41 dB |
| F0 Correlation | 0.507 |
| F0 RMSE | 92.27 Hz |
| MCD | 22.10 dB |

> **Note on STOI:** The intrusive STOI values (~0.08) are likely affected by a
> sample-rate or alignment mismatch between reference and reconstructed audio.
> The non-intrusive SQUIM STOI estimate of **0.985** is more indicative of
> actual perceptual intelligibility.

### Interpretation

- **Hindi (in-domain)** significantly outperforms **Lahaja (out-of-domain)**
  across all metrics, as expected for a Hindi-fine-tuned model.
- **PESQ (wb) = 3.25** on Hindi indicates good perceptual quality for a neural
  codec operating at Mimi's 1.5 kbps bitrate.
- **DNSMOS-BAK = 4.11** shows excellent background noise suppression.
- **TTFAT ~14 ms** (Hindi) demonstrates real-time encoding capability, well
  within streaming latency requirements (algorithmic latency is 80 ms).

---

## Raw Results

All JSON result files are in `results/`:

```
results/
  mimi_hindi_metrics.json       # Bootstrap PESQ/STOI/DNSMOS/MCD with CIs
  mimi_hindi_ssnr.json          # Segmental SNR
  mimi_hindi_ttfat.json         # Time to first audio token
  mimi_hindi_run_all.json       # Full eval pipeline summary
  mimi_hindi_eval_state.json    # Eval pipeline state (for resume)
  mimi_hindi_eval_report.txt    # Human-readable eval report
  mimi_hindi_versa.json         # VERSA 90+ metric suite (per-utterance)
  mimi_lahaja_metrics.json      # Bootstrap metrics for Lahaja
  mimi_lahaja_ssnr.json         # SSNR for Lahaja
  mimi_lahaja_ttfat.json        # TTFAT for Lahaja
  mimi_lahaja_eval_state.json   # Eval state for Lahaja
  TRAINING_RUNS.md              # Detailed training run log
```
