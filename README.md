# codec-finetuning

Benchmark for fine-tuning neural speech codecs on low-resource languages.

Fine-tunes **Mimi**, **DualCodec**, and **Kanade** on Turkish and Hindi with
8 optimizers (AdamW, RAdam, Lion, Prodigy, Schedule-Free, SOAP, Adan, Muon),
WandB Bayesian hyperparameter sweeps, and bootstrap evaluation for error bars.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration Guide](#configuration-guide)
6. [Training](#training)
7. [Optimizer Reference](#optimizer-reference)
8. [Hyperparameter Sweeps](#hyperparameter-sweeps)
9. [Evaluation](#evaluation)
10. [Scaling from Sample to Full](#scaling-from-sample-to-full)
11. [Adding a New Dataset](#adding-a-new-dataset)
12. [Adding a New Codec](#adding-a-new-codec)
13. [Troubleshooting](#troubleshooting)
14. [Citation](#citation)
15. [License](#license)

---

## Quick Start

### Setup + data

```bash
git clone https://github.com/<user>/codec-finetuning.git && cd codec-finetuning
bash scripts/setup.sh                        # installs uv, Python 3.12 venv, all deps
wandb login                                  # paste your WandB API key
bash scripts/download_data.sh turkish_sample # downloads MediaSpeech Turkish (~618 MB)
```

`download_data.sh` automatically downloads and prepares the MediaSpeech Turkish
dataset from OpenSLR (CC BY 4.0). No registration required.

### Train a single run (pipeline validation)

```bash
# Mimi
uv run python train/train_mimi.py --config configs/experiments/mimi_turkish_sample.yaml

# DualCodec
bash train/train_dualcodec.sh --config configs/experiments/dualcodec_turkish_sample.yaml

# Kanade
bash train/train_kanade.sh --config configs/experiments/kanade_turkish_sample.yaml
```

### Run hyperparameter sweeps (full benchmark)

Each codec has its own Bayesian sweep config (40--60 runs, Hyperband early
termination). Set `WANDB_AGENTS` to run multiple trials in parallel on the
same GPU:

```bash
# Mimi -- 96M params, fits 2 agents on a single H100 80 GB
WANDB_AGENTS=2 bash scripts/run_sweep.sh mimi

# DualCodec -- 200M params (includes w2v-bert-2.0), 1 agent per H100
bash scripts/run_sweep.sh dualcodec

# Kanade -- 142M params, fits 2 agents on a single H100 80 GB
WANDB_AGENTS=2 bash scripts/run_sweep.sh kanade
```

On a multi-GPU node you can run more agents. As a rule of thumb:

| GPU | Mimi agents | DualCodec agents | Kanade agents |
|-----|:-----------:|:----------------:|:-------------:|
| 1x H100 80 GB | 2 | 1 | 2 |
| 2x H100 80 GB | 3 | 2 | 3 |
| 1x A100 40 GB | 1 | 1 | 1 |

### Evaluate + analyze

```bash
# Unified evaluation (all stages, auto-resume, results to WandB):
uv run python eval/run_all.py \
    --config configs/experiments/mimi_turkish_sample.yaml \
    --checkpoint outputs/mimi_turkish_sample/best.pt \
    --use-ema

# Extract best config from a completed sweep
uv run python scripts/analyze_sweep.py --sweep-id <entity/project/sweep_id> \
    --output configs/experiments/mimi_turkish_sample_best.yaml
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| GPU | NVIDIA H100 80 GB recommended (tested). A100 40 GB minimum. |
| CUDA drivers | 12.6+ (CUDA 12.6 toolkit is bundled in the PyTorch wheel). |
| Python | 3.12+ |
| OS | Ubuntu 22.04+ (tested on Lambda Stack 24.04 and RunPod PyTorch templates). |
| Accounts | [WandB](https://wandb.ai) (free) for logging and sweeps. [HuggingFace](https://huggingface.co) (free, token needed for the Hindi dataset). |

---

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. The setup
script handles everything end-to-end:

```bash
bash scripts/setup.sh
```

This installs:

- **uv** (if not already present)
- Python 3.12 virtual environment
- PyTorch 2.9.1 + CUDA 12.6
- FlashAttention 2.8.3
- All train and eval dependencies from `pyproject.toml`
- Pretrained codec checkpoints (Mimi, DualCodec, Kanade)
- Whisper large-v3 for WER evaluation

### Manual installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra train --extra eval
uv pip install flash-attn==2.8.3 --no-build-isolation

# VERSA comprehensive evaluation toolkit (installed separately due to a
# protobuf version conflict between versa's s3prl dep and dualcodec's
# descript-audiotools dep):
uv pip install "setuptools<81" \
    "versa-speech-audio-toolkit @ git+https://github.com/shinjiwlab/versa.git"
```

FlashAttention is optional. If it fails to build, training falls back to
PyTorch SDPA automatically. VERSA is required for the comprehensive evaluation
stage in `eval/run_all.py`; if not installed, use `--skip-versa`.

---

## Dataset Preparation

All datasets are resampled to 24 kHz mono and split into speaker-disjoint
train/val/test partitions by `prepare_data.py`.

### Turkish Sample (10 h -- Phase 1 pipeline validation)

MediaSpeech Turkish from [OpenSLR 108](https://www.openslr.org/108/) (CC BY
4.0). The convenience script downloads and prepares the data automatically:

```bash
bash scripts/download_data.sh turkish_sample
```

This downloads `TR.tgz` (~618 MB) from OpenSLR, extracts FLAC audio files with
matching TXT transcripts, resamples to 24 kHz, and creates train/val/test
splits. Or run preparation manually if you already have the data:

```bash
# Extract TR.tgz into data/turkish_sample_raw/
uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml
```

### Hindi (~90 h -- Phase 2, HuggingFace private)

Requires a HuggingFace token with access to the private repository
`tiny-aya-translate/hinglish-casual`.

```bash
huggingface-cli login          # paste your token when prompted
uv run python prepare_data.py --config configs/datasets/hindi.yaml
```

Or via the convenience script:

```bash
bash scripts/download_data.sh hindi
```

### Turkish Full (~100 h -- Phase 2, placeholder)

`configs/datasets/turkish_full.yaml` is a placeholder. Fields marked `TODO`
inside the file should be updated once the full dataset is available. No code
changes are needed -- only the YAML config must be filled in.

---

## Configuration Guide

Configs use a hierarchical **`_bases_` merging** system. Every experiment
config declares a list of base files that are deep-merged in order, with later
values overriding earlier ones.

```
configs/
  base.yaml                    # Shared defaults (optimizer, scheduler, training loop, etc.)
  codecs/
    mimi.yaml                  # Codec-specific: batch size, losses, segment length
    dualcodec.yaml
    kanade.yaml
  datasets/
    turkish_sample.yaml        # Dataset-specific: paths, splits, sample rate
    hindi.yaml
    turkish_full.yaml
  experiments/
    mimi_turkish_sample.yaml   # Composes base + codec + dataset, adds overrides
    ...
  sweeps/
    mimi_sweep.yaml            # WandB sweep parameter space definition
    ...
```

### Example: how merging works

An experiment config such as `mimi_turkish_sample.yaml` declares:

```yaml
_bases_:
  - "../base.yaml"
  - "../codecs/mimi.yaml"
  - "../datasets/turkish_sample.yaml"

training:
  max_steps: 5000              # overrides base.yaml's 10000
```

The config loader (`train/config_loader.py`) recursively resolves and merges
these files. `base.yaml` is applied first, then `mimi.yaml` overlays
codec-specific fields, then `turkish_sample.yaml` overlays dataset-specific
fields, and finally any top-level keys in the experiment file override
everything.

### Overriding values

To change any setting without editing YAML files, create a small override
config and add it to `_bases_`, or modify the experiment YAML directly. All
settings documented in `configs/base.yaml` are available for override.

---

## Training

### Mimi

```bash
uv run python train/train_mimi.py --config configs/experiments/mimi_turkish_sample.yaml
```

### DualCodec

```bash
bash train/train_dualcodec.sh --config configs/experiments/dualcodec_turkish_sample.yaml
```

### Kanade

```bash
bash train/train_kanade.sh --config configs/experiments/kanade_turkish_sample.yaml
```

### Resuming from a checkpoint

```bash
uv run python train/train_mimi.py \
    --config configs/experiments/mimi_turkish_sample.yaml \
    --resume outputs/mimi_turkish_sample/checkpoint_step_3000.pt
```

### Monitoring

Training logs to WandB automatically (controlled by the `wandb` section in
each config). View runs at `https://wandb.ai/<entity>/codec-finetuning`.

Key metrics logged every `log_every` steps: train loss (total, reconstruction,
adversarial, feature matching, commitment), learning rate, gradient norm, and
GPU memory. Reconstructed audio samples are logged every `audio_log_every`
steps.

---

## Optimizer Reference

See [OPTIMIZERS.md](OPTIMIZERS.md) for a comprehensive comparison of all 8
supported optimizers, including when to use each, memory footprint, sweep
parameter interactions, and novel findings potential.

All optimizers are created through a single factory in
`train/optimizer_factory.py`. Key behaviors:

| Optimizer | LR | Scheduler | Warmup | Special |
|---|---|---|---|---|
| AdamW | swept | swept | swept | Baseline |
| RAdam | swept | swept | reduced importance | Stable without warmup |
| Lion | swept (lower range) | swept | swept | Weight decay 3--10x higher |
| Prodigy | auto-estimated | internal | internal | LR-free, set `lr=1.0` |
| Schedule-Free | swept | eliminated | eliminated | Calls `optimizer.eval()` before validation |
| SOAP | swept | swept | swept | Extra `shampoo_beta` param |
| Adan | swept | swept | swept | Three betas (beta1, beta2, beta3) |
| Muon | swept (SGD-scale) | swept (AdamW part) | swept (AdamW part) | Hybrid: Muon for 2-D hidden weights, AdamW for embeddings/norms/biases |

---

## Hyperparameter Sweeps

### 2-Phase strategy

1. **Phase A -- LR Range Finder** (5 runs, ~30 min): Coarse log-uniform LR
   search to identify the viable range for a given optimizer.
2. **Phase B -- Bayesian Sweep** (40--60 runs, Hyperband early termination,
   ~5--8 GPU-hours): Full sweep over LR, weight decay, beta2, scheduler,
   warmup, augmentation preset, segment length, discriminator settings, EMA
   decay, and encoder freezing steps.

Sweep configs live in `configs/sweeps/`. Hyperband kills underperforming runs
early (first bracket at ~step 500) to save compute.

### Launch sweeps

Run all three codecs (adjust `WANDB_AGENTS` for your GPU capacity -- see the
table in [Quick Start](#quick-start)):

```bash
# Mimi (96M params) -- 2 agents on 1x H100 80 GB
WANDB_AGENTS=2 bash scripts/run_sweep.sh mimi

# DualCodec (200M params) -- 1 agent on 1x H100 80 GB
bash scripts/run_sweep.sh dualcodec

# Kanade (142M params) -- 2 agents on 1x H100 80 GB
WANDB_AGENTS=2 bash scripts/run_sweep.sh kanade
```

On 2x H100 or more, increase the agent count:

```bash
WANDB_AGENTS=3 bash scripts/run_sweep.sh mimi
WANDB_AGENTS=2 bash scripts/run_sweep.sh dualcodec
WANDB_AGENTS=3 bash scripts/run_sweep.sh kanade
```

### Analyze results

After each sweep finishes, extract the best configuration as a YAML file:

```bash
uv run python scripts/analyze_sweep.py --sweep-id <entity/project/sweep_id> \
    --output configs/experiments/mimi_turkish_sample_best.yaml

uv run python scripts/analyze_sweep.py --sweep-id <entity/project/sweep_id> \
    --output configs/experiments/dualcodec_turkish_sample_best.yaml

uv run python scripts/analyze_sweep.py --sweep-id <entity/project/sweep_id> \
    --output configs/experiments/kanade_turkish_sample_best.yaml
```

The analysis script prints a ranked table of runs and a parameter importance
summary, then writes the best hyperparameters as a ready-to-use experiment
config.

### Completed sweep: Mimi on Turkish Sample (10 h)

A Bayesian hyperparameter sweep with Hyperband early termination was run on the
10 h Turkish sample dataset to find optimal hyperparameters for Mimi fine-tuning.

- **Sweep ID:** `j7oxgz4p`
- **WandB project:** [cataluna84/codec-finetuning](https://wandb.ai/cataluna84/codec-finetuning)
- **WandB sweep:** [https://wandb.ai/cataluna84/codec-finetuning/sweeps/j7oxgz4p](https://wandb.ai/cataluna84/codec-finetuning/sweeps/j7oxgz4p)
- **Runs:** 68 total across 8 sweep iterations (34 in the main sweep), all 8
  optimizers explored
- **Duration:** Mar 16--18, 2026 (~36 h wall clock on 1x H100 80 GB)
- **Training budget per run:** 5,000 steps

**Top 10 runs by validation loss:**

| Rank | Run ID | Optimizer | LR | Val Loss | Steps |
|:----:|--------|-----------|---:|:--------:|------:|
| 1 | `ooz250pm` | Schedule-Free AdamW | 1.074e-3 | **0.01222** | 5,000 |
| 2 | `d0luhhdd` | Prodigy | 3.82e-4 | 0.01228 | 5,000 |
| 3 | `3kstcode` | SOAP | 1.67e-4 | 0.01228 | 5,000 |
| 4 | `hi2d7mi3` | Adan | 1.2e-5 | 0.01231 | 5,000 |
| 5 | `qymvehel` | Schedule-Free AdamW | 4.8e-5 | 0.01237 | 5,000 |
| 6 | `87see7sz` | AdamW | 1.90e-4 | 0.01243 | 5,000 |
| 7 | `e7z122o8` | Schedule-Free AdamW | 1.5e-5 | 0.01244 | 5,000 |
| 8 | `eu0rw91o` | Schedule-Free AdamW | 9.2e-5 | 0.01245 | 5,000 |
| 9 | `7eejbscw` | Prodigy | 4.4e-5 | 0.01257 | 3,250 |
| 10 | `eo8v05ci` | Prodigy | 2.92e-2 | 0.01259 | 3,250 |

**Key findings:**

- **Schedule-Free AdamW** won the sweep, beating the AdamW baseline (rank 6)
  and taking 4 of the top 8 spots. This validates the NeurIPS 2024 Best Paper
  on a codec fine-tuning task.
- **Prodigy** (LR-free) placed 2nd with no LR tuning, confirming its value for
  fine-tuning workflows.
- **SOAP** (second-order) placed 3rd, supporting the hypothesis that
  second-order information helps on small datasets.
- All 8 optimizers completed runs; the best configuration from `ooz250pm` was
  extracted to `configs/experiments/mimi_turkish_sample_best.yaml` and used as
  the basis for the Phase 2 Hindi fine-tuning run.

---

## Evaluation

### Metrics

| Metric | Range | Direction | Description |
|---|---|---|---|
| PESQ (wideband) | [-0.5, 4.5] | higher is better | Perceptual speech quality |
| STOI | [0, 1] | higher is better | Short-time objective intelligibility |
| DNSMOS (SIG, BAK, OVRL) | [1, 5] | higher is better | Non-intrusive quality via DNS MOS model |
| MCD | 0+ | lower is better | Mel cepstral distortion |
| SSNR | dB | higher is better | Segmental signal-to-noise ratio |
| TTFAT | ms | lower is better | Time to first audio token (streaming latency) |
| WER | [0, 1+] | lower is better | Word error rate via Whisper large-v3 |
| VERSA (90+) | varies | varies | Comprehensive evaluation via [shinjiwlab/versa](https://github.com/shinjiwlab/versa) |

### Bootstrap evaluation (error bars without multi-seed training)

Training is expensive, so instead of running multiple seeds, the benchmark
trains once with seed 42 and computes error bars at evaluation time. The test
set is resampled 20 times with replacement, and each metric is computed per
resample. The reported result is the mean with a 95% bootstrap confidence
interval. This is controlled by the `bootstrap_eval` section in the config.

### Unified evaluation pipeline (recommended)

The recommended way to evaluate is with `eval/run_all.py`, which runs every
evaluation stage in a single command and publishes all metrics to one WandB
run. It has **automatic resume** -- if the pipeline crashes or a stage fails,
re-running the same command picks up from the last completed stage.

```bash
# Full evaluation with EMA checkpoint (all 6 stages):
uv run python eval/run_all.py \
    --config configs/experiments/mimi_hindi.yaml \
    --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \
    --use-ema
```

Stages run in order:

1. **Reconstruction** -- encode/decode test utterances through the codec
2. **SSNR** -- segmental signal-to-noise ratio
3. **TTFAT** -- time to first audio token latency
4. **Bootstrap** -- PESQ, STOI, DNSMOS, MCD with 95% confidence intervals
5. **VERSA** -- comprehensive 90+ metric suite
6. **WandB publish** -- aggregate all results into a single WandB eval run

Results are saved locally to `results/` (JSON + human-readable report) and
published to WandB. A state file (`results/<experiment>_eval_state.json`)
tracks progress so that re-runs skip completed stages automatically.

```bash
# If stage 4 fails, just re-run the same command -- stages 1-3 are cached:
uv run python eval/run_all.py \
    --config configs/experiments/mimi_hindi.yaml \
    --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \
    --use-ema

# Force a clean re-run from scratch:
uv run python eval/run_all.py \
    --config configs/experiments/mimi_hindi.yaml \
    --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \
    --use-ema --restart

# Resume logging to the original training WandB run:
uv run python eval/run_all.py \
    --config configs/experiments/mimi_hindi.yaml \
    --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \
    --use-ema --wandb-run-id iwdd7hfg
```

### Individual evaluation scripts

Each stage can also be run independently if needed:

```bash
# 1. Reconstruct test audio using the EMA model
uv run python eval/reconstruct.py --config configs/experiments/mimi_turkish_sample.yaml --use-ema

# 2. Compute PESQ, STOI, DNSMOS, MCD with bootstrap CIs
uv run python eval/bootstrap_eval.py --experiment mimi_turkish_sample

# 3. Measure streaming latency
uv run python eval/measure_ttfat.py --config configs/experiments/mimi_turkish_sample.yaml

# 4. Measure segmental SNR
uv run python eval/measure_ssnr.py --experiment mimi_turkish_sample

# 5. Run VERSA comprehensive evaluation
bash eval/run_versa.sh mimi_turkish_sample

# 6. Log all results to the WandB run
uv run python eval/log_to_wandb.py --experiment mimi_turkish_sample
```

### Full pipeline (train + evaluate in one command)

```bash
bash scripts/run_all.sh mimi turkish_sample
```

This runs training, reconstruction, bootstrap evaluation, TTFAT, SSNR, and
WandB logging in sequence.

---

## Scaling from Sample to Full

The benchmark is designed in two phases:

- **Phase 1** uses the 10-hour Turkish sample (MediaSpeech) for end-to-end
  pipeline validation.
- **Phase 2** scales to ~100 h Turkish and ~90 h Hindi.

The transition requires **only config changes** -- zero code modifications:

1. Fill in the `TODO` fields in `configs/datasets/turkish_full.yaml` with the
   actual dataset paths and statistics.
2. Run data preparation:
   ```bash
   uv run python prepare_data.py --config configs/datasets/turkish_full.yaml
   ```
3. Train with the corresponding experiment config:
   ```bash
   uv run python train/train_mimi.py --config configs/experiments/mimi_turkish_full.yaml
   ```

The full-dataset experiment configs (`*_turkish_full.yaml`, `*_hindi.yaml`)
already exist in `configs/experiments/` and reference the correct dataset and
codec configs.

---

## Adding a New Dataset

1. **Create a dataset config** at `configs/datasets/<name>.yaml`. Copy an
   existing config (e.g., `turkish_sample.yaml`) as a template and update the
   source, paths, column mappings, and split ratios.

2. **Create experiment configs** for each codec:
   ```
   configs/experiments/mimi_<name>.yaml
   configs/experiments/dualcodec_<name>.yaml
   configs/experiments/kanade_<name>.yaml
   ```
   Each should reference `../base.yaml`, the appropriate codec config, and
   your new dataset config in `_bases_`.

3. **Run data preparation:**
   ```bash
   uv run python prepare_data.py --config configs/datasets/<name>.yaml
   ```

4. **Train:**
   ```bash
   uv run python train/train_mimi.py --config configs/experiments/mimi_<name>.yaml
   ```

---

## Adding a New Codec

1. **Create a codec config** at `configs/codecs/<name>.yaml`. Define the
   pretrained model, sample rate, frame rate, codebook count, micro batch size,
   loss configuration, and any codec-specific parameters.

2. **Create a training script** (or shell wrapper):
   - Python: `train/train_<name>.py`
   - Shell: `train/train_<name>.sh`

3. **Add reconstruction support** in `eval/reconstruct.py` so the evaluation
   pipeline can decode tokens back to waveforms for the new codec.

4. **Register the codec in the sweep agent** (`train/sweep_agent.py`) to
   enable hyperparameter sweeps.

5. **Create experiment configs** for each dataset:
   ```
   configs/experiments/<name>_turkish_sample.yaml
   configs/experiments/<name>_hindi.yaml
   ```

---

## Troubleshooting

**CUDA out of memory** -- Reduce `codec.training.micro_batch_size` in the
codec config (e.g., `configs/codecs/mimi.yaml`). The effective batch size
equals `micro_batch_size * training.grad_accum_steps`, so you can increase
`grad_accum_steps` to compensate.

**WandB authentication** -- Run `wandb login` and paste your API key. The
`wandb.enabled` flag in the config can be set to `false` to disable logging
entirely during debugging.

**HuggingFace authentication** -- Run `huggingface-cli login` with a token
that has access to the private Hindi dataset repository.

**FlashAttention fails to install** -- This is non-fatal. Training
automatically falls back to PyTorch SDPA (Scaled Dot-Product Attention). You
may see a warning at startup; it can be safely ignored.

**NaN loss during training** -- The training loop handles NaN loss
automatically: it halves the learning rate and reloads the last checkpoint
after 3 consecutive NaN steps. If the problem persists, the optimizer or
learning rate may be unsuitable for the codec/dataset combination.

**Slow training** -- Verify that `torch.compile` is enabled
(`training.compile: true` in the config). Check that FlashAttention is
available (logged at startup). On H100, ensure bf16 precision is active
(`training.precision: "bf16"`).

**Data preparation fails** -- Ensure the raw audio is in the correct directory
(`dataset.local_raw_dir` in the dataset config). Check that audio files are
readable WAV or FLAC. The script logs skipped files with reasons.

---

## Citation

```bibtex
@software{codec_finetuning,
  title   = {codec-finetuning: Benchmark for Fine-tuning Neural Speech Codecs
             on Low-Resource Languages},
  year    = {2026},
  url     = {https://github.com/<user>/codec-finetuning},
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
