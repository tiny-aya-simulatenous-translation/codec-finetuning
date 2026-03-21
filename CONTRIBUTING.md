# Contributing to codec-finetuning

Thank you for your interest in contributing! This document covers code style,
how to extend the benchmark (datasets, codecs, optimizers, metrics), and the
pull-request process.

---

## Code Style

| Rule | Detail |
|------|--------|
| Standard | PEP-8 |
| Linter | **ruff** (Google convention) |
| Line length | 100 characters max |
| Docstrings | Google-style on **all** functions |
| Type hints | Required on all function signatures |
| License header | MIT license header in every module docstring |

Run the linter before submitting any changes:

```bash
uv run ruff check .
```

---

## Adding a New Dataset

### Checklist

- [ ] **Dataset config** at `configs/datasets/<name>.yaml`
- [ ] **Experiment configs** for each codec at
      `configs/experiments/{codec}_{dataset}.yaml`
- [ ] **Data preparation** tested via `prepare_data.py`
- [ ] **Test split** has a `manifest.json` that the eval pipeline can read

### Steps

1. **Copy the template config:**
   ```bash
   cp configs/datasets/turkish_sample.yaml configs/datasets/your_dataset.yaml
   ```

2. **Fill in required fields:**
   - `name` -- unique identifier for the dataset
   - `language` -- ISO 639-1 code (e.g. `hi`, `tr`)
   - `source` -- `local` or `huggingface`
   - `column_mappings` -- map source columns to expected schema
   - `splits` -- train / validation / test definitions
   - `filters` -- any filtering criteria (duration, quality, etc.)

3. **Create experiment configs** for each codec:
   ```
   configs/experiments/{codec}_{dataset}.yaml
   ```

4. **Test data preparation:**
   ```bash
   uv run python prepare_data.py --config configs/datasets/your_dataset.yaml
   ```

5. **Verify the eval pipeline runs end-to-end** (at least with `--skip-versa`
   for speed):
   ```bash
   uv run python eval/run_all.py \
       --config configs/experiments/{codec}_{dataset}.yaml \
       --skip-wandb --skip-versa
   ```

---

## Adding a New Codec

### Architecture overview

The evaluation pipeline uses a **codec registry** (`eval/codec_registry.py`)
as its single extension point. Each codec registers a `CodecEvalHooks`
dataclass with three things:

1. **`load`** -- how to load the pretrained model and apply a checkpoint
2. **`encode_decode`** -- how to round-trip a waveform through the codec
3. **`extra_metrics`** (optional) -- codec-specific evaluation stages that
   run automatically alongside the standard pipeline

The rest of the pipeline (SSNR, TTFAT, bootstrap, VERSA, WandB publish) is
codec-agnostic -- it operates on reconstructed WAV files.

### Checklist

- [ ] **Codec config** at `configs/codecs/<name>.yaml` (pretrained model,
      sample rate, frame rate, codebook count, batch size, losses)
- [ ] **Training script** at `train/train_<name>.py` (or `.sh`)
- [ ] **Eval hooks** registered in `eval/codec_registry.py`:
  - [ ] `load` function (load pretrained + apply checkpoint with
        `_load_checkpoint_state` helper)
  - [ ] `encode_decode` function (encode waveform to tokens, decode back)
  - [ ] `extra_metrics` (optional) -- list of `(name, fn)` tuples for any
        codec-specific metrics
- [ ] **Sweep agent** branch in `train/sweep_agent.py`
- [ ] **Experiment configs** for each dataset:
      `configs/experiments/<name>_{dataset}.yaml`
- [ ] **Sweep config** at `configs/sweeps/<name>_sweep.yaml`
- [ ] **Dependency** added to `pyproject.toml` if the codec needs an
      external package
- [ ] Tested on the Turkish sample dataset (10 h, fast iteration)

### Steps

1. **Create a codec config:**
   ```
   configs/codecs/your_codec.yaml
   ```
   Include architecture details, pretrained model path, and loss definitions.

2. **Create a training script:**
   ```
   train/train_your_codec.py
   ```
   Use `train/train_mimi.py` as the reference implementation.

3. **Register eval hooks** in `eval/codec_registry.py`. Add your codec's
   functions alongside the existing Mimi/DualCodec/Kanade implementations
   and add a new `CodecEvalHooks` entry in `_builtin_codecs()`:

   ```python
   # -- YourCodec -----------------------------------------------------------

   def _load_your_codec(config, checkpoint, use_ema, device):
       import your_codec_package
       model = your_codec_package.load_model(config["codec"]["pretrained"])
       if checkpoint is not None:
           sd = _load_checkpoint_state(checkpoint, use_ema, device)
           if sd is not None:
               model.load_state_dict(sd, strict=False)
       return model.to(device).eval()


   def _encode_decode_your_codec(model, waveform):
       tokens = model.encode(waveform)
       return model.decode(tokens)
   ```

   Then add to `_builtin_codecs()`:

   ```python
   CodecEvalHooks(
       name="your_codec",
       load=_load_your_codec,
       encode_decode=_encode_decode_your_codec,
       # Optional: codec-specific metrics
       extra_metrics=[
           ("my_metric", _compute_my_metric),
       ],
   ),
   ```

   The optional `extra_metrics` callables receive `(config, experiment,
   results_dir)` and return a JSON-serialisable dict. They are run as
   additional stages in `eval/run_all.py` between VERSA and WandB publish,
   with full state-tracking (resume on failure).

4. **Wire up sweeps** -- add a codec branch to `train/sweep_agent.py`.

5. **Create experiment and sweep configs:**
   ```
   configs/experiments/{your_codec}_{dataset}.yaml
   configs/sweeps/{your_codec}_sweep.yaml
   ```

6. **Verify evaluation runs end-to-end:**
   ```bash
   uv run python eval/run_all.py \
       --config configs/experiments/{your_codec}_{dataset}.yaml \
       --skip-wandb --skip-versa
   ```

7. **Test on the Turkish sample first** (10 h, fast iteration) before scaling
   to larger datasets like Hindi.

### What you do NOT need to change

Because the pipeline is registry-driven, adding a new codec requires **no
changes** to:

- `eval/reconstruct.py` -- delegates to the registry
- `eval/run_all.py` -- dynamically discovers codec-specific stages
- `eval/measure_ssnr.py` -- codec-agnostic (operates on WAV files)
- `eval/measure_ttfat.py` -- uses `reconstruct.load_model` which dispatches
  via the registry
- `eval/bootstrap_eval.py` -- codec-agnostic
- `eval/run_versa.sh` -- codec-agnostic
- `eval/log_to_wandb.py` -- codec-agnostic

---

## Adding a New Optimizer

### Checklist

- [ ] **Registered** in `train/optimizer_factory.py`
- [ ] **Documented** in `OPTIMIZERS.md` (comparison table + sweep notes)
- [ ] **Enabled** in sweep configs (`configs/sweeps/*.yaml`)
- [ ] **Special behavior** handled in `train/sweep_agent.py`
- [ ] **Dependency** added to `pyproject.toml`

### Steps

1. **Register the optimizer** in `train/optimizer_factory.py`:
   - Import the optimizer class
   - Add a creation function
   - Define parameter-group logic (e.g. separate LR for encoder vs. decoder)

2. **Document it** in `OPTIMIZERS.md`:
   - Add a row to the comparison table
   - Note sweep considerations and memory footprint

3. **Enable it in sweep configs** -- add the optimizer name to the
   `optimizer.values` list in the relevant sweep YAML files.

4. **Handle special behavior** in `train/sweep_agent.py` if needed (e.g.
   Prodigy requires disabling the LR scheduler).

5. **Add the dependency** to `pyproject.toml` under
   `[project.optional-dependencies.train]`.

---

## Adding New Evaluation Metrics

### Codec-agnostic metrics (applies to all codecs)

1. Implement the metric computation in `eval/bootstrap_eval.py`.
2. Update `eval/log_to_wandb.py` to report the new metric.
3. If the metric requires a new pipeline stage, add it to `eval/run_all.py`:
   wrap the computation in a `_run_<stage>()` helper, add the stage to the
   orchestrator with state tracking (`state.is_done()` / `state.mark_done()`),
   and publish results in `_publish_to_wandb()`.

### Codec-specific metrics (only applies to one codec)

Use the `extra_metrics` field in `CodecEvalHooks` (see "Adding a New Codec"
above). This avoids polluting the shared pipeline with codec-specific logic
and automatically gets state-tracking and WandB publishing.

```python
def _compute_my_codec_metric(config, experiment, results_dir):
    """Return a dict of metric_name -> value."""
    # Your metric computation here
    return {"my_metric_mean": 0.95, "my_metric_std": 0.02}

CodecEvalHooks(
    name="my_codec",
    load=_load_my_codec,
    encode_decode=_encode_decode_my_codec,
    extra_metrics=[("my_metric", _compute_my_codec_metric)],
)
```

### Unified evaluation pipeline

All evaluation stages are orchestrated by `eval/run_all.py`. This script:

- Runs reconstruction, SSNR, TTFAT, bootstrap metrics (PESQ/STOI/DNSMOS/MCD),
  VERSA, and any codec-specific extra metrics in sequence.
- Publishes all results to a single WandB eval run.
- Supports **automatic resume** via a persistent state file -- re-running the
  same command after a crash skips already-completed stages.
- Accepts `--restart` to force a clean re-run from scratch.
- Dynamically discovers and runs codec-specific metric stages from the codec
  registry.

---

## PR Checklist

Before submitting a pull request, make sure **all** of the following are true:

### General

- [ ] `uv run ruff check .` passes
- [ ] All new functions have Google-style docstrings with type hints
- [ ] New config files have inline comments on every field
- [ ] Tested on the Turkish sample dataset (10 h, fast iteration)
- [ ] Updated `README.md` if adding user-facing features
- [ ] No secrets, API keys, or HuggingFace tokens in committed files

### If adding a codec

- [ ] Codec config at `configs/codecs/<name>.yaml`
- [ ] Training script at `train/train_<name>.py`
- [ ] Eval hooks registered in `eval/codec_registry.py` (load + encode_decode)
- [ ] Sweep agent branch in `train/sweep_agent.py`
- [ ] Experiment configs for at least one dataset
- [ ] `eval/run_all.py` runs end-to-end without errors

### If adding a dataset

- [ ] Dataset config at `configs/datasets/<name>.yaml`
- [ ] Experiment configs for at least one codec
- [ ] `prepare_data.py` produces a valid manifest with train/val/test splits
- [ ] `eval/run_all.py` runs end-to-end on the test split

### If adding a metric

- [ ] Metric implemented in `eval/bootstrap_eval.py` (codec-agnostic) or
      registered via `extra_metrics` in `eval/codec_registry.py`
      (codec-specific)
- [ ] Results published to WandB in `eval/run_all.py` or via the registry's
      automatic codec-extras publishing
