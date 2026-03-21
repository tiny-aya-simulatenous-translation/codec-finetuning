# Contributing to codec-finetuning

Thank you for your interest in contributing! This document covers code style,
documentation conventions, the evaluation pipeline architecture, how to extend
the benchmark (datasets, codecs, optimizers, metrics), and the pull-request
process.

---

## Code Style

| Rule | Detail |
|------|--------|
| Standard | PEP-8 |
| Linter | **Ruff** (with Google docstring convention) |
| Line length | 100 characters max |
| Docstrings | Google-style on **all** public functions and classes |
| Type hints | Required on all function signatures |
| License header | MIT license header in every module docstring |

Run the linter before submitting any changes:

```bash
uv sync --extra all                   # install everything (train + eval + dev)
uv run ruff check .                  # lint all files
uv run ruff check --fix .            # auto-fix safe violations
```

> **Note:** Ruff is in the `dev` optional dependency group.  The recommended
> way to install everything at once is `uv sync --extra all`, which pulls in
> `train`, `eval`, and `dev` in one shot.  See [agents.md](agents.md) for the
> full dependency rules and documentation standards.

### Why Ruff?

[Ruff](https://docs.astral.sh/ruff/) is an extremely fast Python linter and
formatter written in Rust by [Astral](https://astral.sh/) (the same team that
builds `uv`, the package manager used in this project).  It replaces multiple
tools -- flake8, isort, pycodestyle, pyflakes, pydocstyle, and pyupgrade --
with a single binary that runs 10-100x faster than any of them individually.

**Why it matters for contributors:**

- **Speed** -- Ruff checks the entire codebase in under 1 second, so there is
  no excuse not to run it before every commit.  Traditional linters (flake8 +
  isort + pydocstyle) take 5-10 seconds on a project this size.
- **Single tool** -- Instead of configuring and running flake8, isort, and
  pydocstyle separately, Ruff handles all of them from one config block in
  `pyproject.toml`.  Fewer tools means fewer version conflicts and simpler CI.
- **Unified config** -- Everything lives in `pyproject.toml` under
  `[tool.ruff]`.  No `.flake8`, `.isort.cfg`, or `.pydocstyle` files to
  maintain.
- **Auto-fixable** -- Many Ruff rules (import sorting, unused imports,
  deprecated syntax) can be auto-fixed with `ruff check --fix`.

### Enabled rule sets

The project enables the following Ruff rule sets (configured in
`pyproject.toml` under `[tool.ruff.lint]`):

| Code | Rule set | What it catches |
|------|----------|-----------------|
| `E` | pycodestyle errors | Syntax errors, indentation, whitespace |
| `F` | pyflakes | Unused imports, undefined names, redefined variables |
| `W` | pycodestyle warnings | Trailing whitespace, blank lines, line length |
| `I` | isort | Import ordering (stdlib, third-party, local) |
| `D` | pydocstyle | Missing or malformed docstrings |
| `UP` | pyupgrade | Deprecated syntax (e.g. `dict()` -> `{}`, old-style type hints) |

### Google docstring convention

The `D` (pydocstyle) rules are configured with `convention = "google"` in
`[tool.ruff.lint.pydocstyle]`.  This enforces the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
docstring format, which uses indented section headers like `Args:`,
`Returns:`, `Raises:`, and `Example:` rather than the numpy-style
underline-based sections or plain PEP 257.

Google-style was chosen because:

- It is **more compact** than numpy-style (no underlines, fewer blank lines),
  which matters when every function needs a docstring.
- It is the **most widely understood** format in the ML/audio research
  community (PyTorch, JAX, and TensorFlow all use it).
- It renders cleanly in **IDEs** (VS Code, PyCharm) and **Sphinx** (via the
  `napoleon` extension) without additional configuration.

Example of a correctly formatted Google-style docstring:

```python
def compute_ssnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_rate: int,
    segment_ms: float = 25.0,
) -> float:
    """Compute Segmental Signal-to-Noise Ratio for one utterance.

    Segments the waveform into overlapping frames, computes per-frame
    SNR, and returns the mean across non-silent frames.

    Args:
        original: Reference audio as a 1-D numpy array.
        reconstructed: Degraded audio as a 1-D numpy array.
        sample_rate: Audio sample rate in Hz.
        segment_ms: Frame length in milliseconds.

    Returns:
        Mean SSNR in dB.

    Raises:
        ValueError: If the arrays have incompatible shapes.
    """
```

---

## Documentation Conventions

Every Python file in this project follows a consistent documentation pattern.
Please maintain this standard in all contributions.

### Module-level docstrings

Every `.py` file must start with a module docstring that includes:

1. **One-line summary** -- what the module does.
2. **Extended description** -- how it works, key algorithms.
3. **Performance section** (where applicable) -- before/after timings,
   technique used (e.g. batching, multiprocessing), measured on which
   hardware.
4. **Pipeline position** (for eval modules) -- which stage number this is
   in the `run_all.py` pipeline, and what it depends on.
5. **Usage example** -- a runnable `uv run` command.
6. **License** -- `License: MIT`.

Example (from `eval/bootstrap_eval.py`):

```python
"""Bootstrap resampling evaluation for error bars.

Computes per-utterance quality metrics (PESQ, STOI, DNSMOS, MCD)...

Performance
-----------
**Parallel per-utterance metric computation** (added 2026-03-21):
    Before: ~42 min (sequential)
    After:  ~3 min  (16-worker process pool)

Pipeline position
-----------------
This module is **Stage 4** of the unified evaluation pipeline.

Usage::
    uv run python eval/bootstrap_eval.py --experiment mimi_hindi

License: MIT
"""
```

### Function/method docstrings

Use Google-style with `Args`, `Returns`, `Raises` sections:

```python
def compute_ssnr(original: np.ndarray, reconstructed: np.ndarray, ...) -> float:
    """Compute Segmental Signal-to-Noise Ratio for one utterance.

    Args:
        original: Original audio as a 1-D numpy array.
        reconstructed: Reconstructed audio as a 1-D numpy array.

    Returns:
        Mean SSNR in dB across non-silent segments.

    Raises:
        ValueError: If arrays have incompatible shapes.
    """
```

### Inline comments

Add comments for:
- **Non-obvious math** (e.g. MCD formula, SNR calculation).
- **Performance-critical decisions** (e.g. why `chunksize=32`, why sort by
  length before batching).
- **Worker functions** -- explain they are pool workers and why they take a
  tuple argument.
- **Module-level constants** -- explain what they control and valid ranges.

### Config YAML files

Every YAML config field should have an inline comment explaining its purpose,
valid range, and default value.  See `configs/base.yaml` for the reference
style.

---

## Evaluation Pipeline Architecture

The unified evaluation pipeline (`eval/run_all.py`) orchestrates 6 stages
with automatic resume, concurrent execution, and parallel computation.
Understanding this architecture is essential for contributing evaluation
changes.

### Stage execution flow

```
Stage 1: Reconstruction (GPU, batched)
    ↓
Stage 2: SSNR ─────────┐
                        ├── concurrent (ThreadPoolExecutor, 2 workers)
Stage 3: TTFAT ─────────┘   SSNR=CPU-only, TTFAT=GPU-only → zero contention
    ↓
Stage 4: Bootstrap (CPU, ProcessPoolExecutor, 16 workers)
    ↓
Stage 5: VERSA (sharded, 4 parallel subprocesses)
    ↓
Stage 6: WandB Publish
```

### Parallelisation techniques

| Stage | Technique | Implementation | Why |
|-------|-----------|----------------|-----|
| Reconstruction | Batched GPU inference | Sort by length, pad per batch, single `encode_decode()` call | Eliminates per-utterance GPU idle time |
| SSNR | `ProcessPoolExecutor` | Top-level `_compute_ssnr_for_entry()` worker, `chunksize=32` | Embarrassingly parallel, CPU-only numpy |
| SSNR + TTFAT | `ThreadPoolExecutor` | 2-thread stage-level overlap | CPU vs GPU, zero resource contention |
| Bootstrap | `ProcessPoolExecutor` | Top-level `_compute_metrics_worker()`, `chunksize=8` | Largest bottleneck, 4 independent metrics per utterance |
| VERSA | Sharded subprocesses | Manifest split into N scp files, N parallel `run_versa.sh` via `ThreadPoolExecutor` | External tool, can't parallelise internally |

### State management

The `EvalState` class persists progress to
`results/<experiment>_eval_state.json` after every stage.  A fingerprint hash
of `(experiment, checkpoint, split, use_ema)` detects config changes and
invalidates stale state.  Each stage can be in one of three states:

- `ok` -- completed successfully, result cached.
- `failed` -- error recorded, will be retried on next run.
- (absent) -- not yet attempted.

### Adding a new evaluation stage

1. Write a `_run_<stage>()` helper function in `run_all.py`.
2. Add the stage to the orchestrator between the appropriate existing stages.
3. Wrap it with `state.is_done()` / `state.mark_done()` /
   `state.mark_failed()` for resume support.
4. Publish its results in `_publish_to_wandb()`.
5. Add the stage name to `STAGE_NAMES` for the failure summary.

### Worker function conventions

Process pool workers must be **top-level functions** (not lambdas, closures,
or methods) because `ProcessPoolExecutor` uses `pickle` for IPC.  They accept
a single tuple argument because `Executor.map()` requires a single-argument
callable.

```python
def _compute_ssnr_for_entry(
    args: Tuple[str, str, int, float, float],
) -> float:
    """Process-pool worker: compute SSNR for a single utterance pair."""
    ref_path, deg_path, sample_rate, segment_ms, overlap_ms = args
    ...
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

### Eval-only datasets (no training split)

Some datasets (e.g. Lahaja) are evaluation-only and have no training split.
For these:

1. Write a dedicated prep script under `scripts/` (see
   `scripts/prepare_lahaja.py` as the reference implementation).
2. Only create the `data/<name>/test/` directory with a `manifest.json`.
3. Set `train_ratio: 0.0` and `val_ratio: 0.0` in the dataset config.
4. The experiment config still needs training hyperparameters (they are
   required by `config_loader.py` validation) but they will not be used --
   copy them from the checkpoint's original training config.
5. Use `torchcodec` (via the HuggingFace `datasets` Audio feature) for audio
   decoding and `torchaudio` for writing WAV files -- do not use `soundfile`.

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
