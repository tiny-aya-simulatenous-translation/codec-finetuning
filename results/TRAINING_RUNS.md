# Training Runs Log

All runs are tracked in the WandB project:
**[cataluna84/codec-finetuning](https://wandb.ai/cataluna84/codec-finetuning)**

Hardware: 1x NVIDIA H100 PCIe 80 GB, Ubuntu 22.04, CUDA 12.8, Python 3.12,
PyTorch 2.9.1.

---

## Phase 1: Hyperparameter Sweep -- Mimi on Turkish Sample (10 h)

**Goal:** Find optimal hyperparameters for Mimi codec fine-tuning across 8
optimizers using Bayesian optimization with Hyperband early termination.

| Field | Value |
|---|---|
| Sweep ID | `j7oxgz4p` |
| WandB sweep | [wandb.ai/cataluna84/codec-finetuning/sweeps/j7oxgz4p](https://wandb.ai/cataluna84/codec-finetuning/sweeps/j7oxgz4p) |
| Codec | Mimi (kyutai/mimi, 96M params) |
| Dataset | Turkish Sample (MediaSpeech, 10 h, OpenSLR 108) |
| Optimizers | AdamW, RAdam, Lion, Prodigy, Schedule-Free AdamW, SOAP, Adan, Muon |
| Total runs | 68 (across 8 sweep iterations, 34 in main sweep) |
| Steps per run | 5,000 (fixed budget) |
| Duration | Mar 16--18, 2026 (~36 h wall clock) |
| Method | Bayesian (WandB), Hyperband early termination (eta=3, s=2, min_iter=500) |

### Earlier sweep iterations (pipeline validation)

These sweeps were run during development to validate the sweep pipeline and
debug configuration issues. Their results informed the final main sweep.

| Sweep ID | Runs | Notes |
|---|---:|---|
| `hblooqc6` | 6 | Initial pipeline test |
| `7h904d6b` | 10 | First multi-optimizer run |
| `bkglf87d` | 5 | Scheduler exploration |
| `232hw2za` | 5 | SOAP/Adan testing |
| `xzdokt3y` | 1 | Lion single-run test |
| `24gfpbgw` | 2 | Muon/AdamW pair test |
| `lw507k7t` | 5 | Novel optimizers focus |

### Main sweep results (`j7oxgz4p`, 34 runs)

| Rank | Run ID | WandB | Optimizer | LR | Val Loss | Steps |
|:----:|--------|-------|-----------|---:|:--------:|------:|
| 1 | `ooz250pm` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/ooz250pm) | Schedule-Free AdamW | 1.074e-3 | **0.01222** | 5,000 |
| 2 | `d0luhhdd` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/d0luhhdd) | Prodigy | 3.82e-4 | 0.01228 | 5,000 |
| 3 | `3kstcode` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/3kstcode) | SOAP | 1.67e-4 | 0.01228 | 5,000 |
| 4 | `hi2d7mi3` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/hi2d7mi3) | Adan | 1.2e-5 | 0.01231 | 5,000 |
| 5 | `qymvehel` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/qymvehel) | Schedule-Free AdamW | 4.8e-5 | 0.01237 | 5,000 |
| 6 | `87see7sz` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/87see7sz) | AdamW | 1.90e-4 | 0.01243 | 5,000 |
| 7 | `e7z122o8` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/e7z122o8) | Schedule-Free AdamW | 1.5e-5 | 0.01244 | 5,000 |
| 8 | `eu0rw91o` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/eu0rw91o) | Schedule-Free AdamW | 9.2e-5 | 0.01245 | 5,000 |
| 9 | `7eejbscw` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/7eejbscw) | Prodigy | 4.4e-5 | 0.01257 | 3,250 |
| 10 | `eo8v05ci` | [view](https://wandb.ai/cataluna84/codec-finetuning/runs/eo8v05ci) | Prodigy | 2.92e-2 | 0.01259 | 3,250 |

### Best hyperparameters (from `ooz250pm`)

Extracted to `configs/experiments/mimi_turkish_sample_best.yaml`:

```yaml
optimizer:
  name: schedulefree_adamw
  lr: 1.074e-3
  betas: [0.9, 0.95]
  weight_decay: 0.0
scheduler:
  name: constant          # disabled by Schedule-Free internals
  warmup_steps: 0         # disabled by Schedule-Free internals
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

### Key findings

- **Schedule-Free AdamW** won the sweep and took 4 of the top 8 spots.
- **Prodigy** (LR-free) placed 2nd with no LR tuning required.
- **SOAP** (second-order) placed 3rd, supporting the small-data hypothesis.
- **Adan** (triple-momentum) placed 4th, validating its CNN+transformer+GAN
  design on a real codec architecture.
- **AdamW** baseline placed 6th -- all top 5 are non-standard optimizers.
- **Muon** completed runs but ranked lower (11th, 13th), partially confirming
  the author's open question about fine-tuning applicability. First known test
  of Muon on codec/audio/GAN training.
- No augmentation (`preset: none`) was best, suggesting the 10 h dataset is
  clean enough that augmentation hurts more than it helps at this scale.
- `beta2=0.95` (lower than default 0.999) was preferred, consistent with GAN
  training literature.

---

## Phase 2: Fine-tuning -- Mimi on Hindi (~90 h)

**Goal:** Full-scale fine-tuning of Mimi on the Hindi conversational speech
dataset using the best hyperparameters from the Phase 1 sweep.

| Field | Value |
|---|---|
| Run name | `mimi-hindi-sweep-best` |
| Run ID | `iwdd7hfg` |
| WandB run | [wandb.ai/cataluna84/codec-finetuning/runs/iwdd7hfg](https://wandb.ai/cataluna84/codec-finetuning/runs/iwdd7hfg) |
| Config | `configs/experiments/mimi_hindi.yaml` |
| Codec | Mimi (kyutai/mimi, 96M params) |
| Dataset | Hindi (~90 h, 33,275 utterances, tiny-aya-translate/hinglish-casual) |
| Optimizer | Schedule-Free AdamW (lr=1.074e-3, betas=[0.9, 0.95], wd=0.0) |
| Precision | bf16 |
| Grad accumulation | 8 steps |
| Max steps | 50,000 (scaled ~10x from sweep's 5,000 for ~9x more data) |
| Duration | Mar 18 17:49 -- Mar 19 00:46 UTC (~7 h) |
| Status | **Completed successfully** |

### Training progression

| Steps | Val Loss | Phase |
|------:|:--------:|-------|
| 1,000 | 0.0216 | Rapid improvement |
| 5,000 | 0.0206 | |
| 10,000 | 0.0204 | Diminishing returns begin |
| 15,000 | 0.0203 | |
| 22,000 | 0.0201 | Plateau begins |
| 39,000 | 0.0200 | Best observed |
| 50,000 | 0.0200 | Final |

### Final metrics (WandB summary)

| Metric | Value |
|---|---|
| val/loss | 0.02001 |
| train/reconstruction | 0.02015 |
| train/adversarial | 0.01104 |
| train/feature_matching | 0.03965 |
| train/lr | 0.001074 |

### Checkpoints

20 checkpoints saved every 2,500 steps in `outputs/mimi_hindi/`:

```
checkpoint_step_2500.pt   ...   checkpoint_step_50000.pt
```

Each checkpoint is ~1 GB and contains `model_state_dict`, `ema_state_dict`,
`optimizer_state_dict`, and `scheduler_state_dict`.

### Observations

- **Fully converged.** Val loss decreased monotonically from 0.0216 to 0.0200
  with no instability, NaN losses, or OOM errors.
- **Saturated after ~25k steps.** The last 25k steps yielded only 0.0001
  improvement -- training was more than adequate.
- **torch.compile recompilation warnings** at startup are benign (standard
  `torch._dynamo` guard mismatches for varying tensor sizes) and did not recur.
- **Early stopping** was configured (patience=10) but never triggered since
  val loss kept slowly improving.

---

## Phase 3: Evaluation (pending)

**Goal:** Comprehensive evaluation of the fine-tuned Mimi Hindi checkpoint
using the unified evaluation pipeline (`eval/run_all.py`).

| Field | Value |
|---|---|
| Config | `configs/experiments/mimi_hindi.yaml` |
| Checkpoint | `outputs/mimi_hindi/checkpoint_step_50000.pt` |
| EMA weights | Yes |
| Eval split | test (1,665 utterances) |

### Command

```bash
uv run python eval/run_all.py \
    --config configs/experiments/mimi_hindi.yaml \
    --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \
    --use-ema
```

### Stages

1. **Reconstruction** -- encode/decode 1,665 test utterances through the
   fine-tuned codec
2. **SSNR** -- segmental signal-to-noise ratio
3. **TTFAT** -- time to first audio token latency (50 timed runs, 5 warmup)
4. **Bootstrap** -- PESQ (wb/nb), STOI, DNSMOS (sig/bak/ovrl), MCD with 95%
   confidence intervals (20 bootstrap resamples)
5. **VERSA** -- comprehensive 90+ metric suite via shinjiwlab/versa
6. **WandB publish** -- all results aggregated into a single eval run

Results will be saved locally to `results/` and published to WandB.
The pipeline supports automatic resume -- if any stage fails, re-running the
same command picks up where it left off.
