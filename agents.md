# Agent Guidelines for codec-finetuning

This document is for AI coding agents (and human contributors who think like
one).  It codifies the dependency rules, documentation standards, linting
requirements, and design decisions that were applied across every file in this
repository.  Treat it as the authoritative checklist before touching any code.

---

## 1. Dependency Rules

### 1.1 Where dependencies are declared

| Scope | Location in `pyproject.toml` | Install command |
|-------|------------------------------|-----------------|
| Core (always installed) | `[project] dependencies` | `uv sync` |
| Training only | `[project.optional-dependencies] train` | `uv sync --extra train` |
| Evaluation only | `[project.optional-dependencies] eval` | `uv sync --extra eval` |
| Development / linting | `[project.optional-dependencies] dev` | `uv sync --extra dev` |
| **Everything** | `[project.optional-dependencies] all` | **`uv sync --extra all`** |
| Transitive overrides | `[tool.uv] override-dependencies` | (applied automatically) |

> **Recommended for contributors:** `uv sync --extra all` installs train +
> eval + dev in one command.  The individual extras exist for CI flexibility
> (e.g. a lint-only job needs only `--extra dev`).

### 1.2 Key dependency decisions

| Dependency | Version | Why pinned / capped | Notes |
|------------|---------|---------------------|-------|
| `torch` | `>=2.9.1` | CUDA 12.6 wheel from `pytorch-cu126` index | Sourced via `[tool.uv.sources]` |
| `torchcodec` | `==0.9.0` | Required by `datasets>=4.7` for audio decoding; replaces `soundfile` | In core deps, sourced from `pytorch-cu126` index |
| `protobuf` | `>=4.21.1,<7` | wandb 0.25.x only has shims for protobuf v4/v5/v6; v7 breaks `wandb_telemetry_pb2.Imports` | Override in `[tool.uv]` |
| `wandb` | `>=0.19.0` | Lockfile should resolve to `>=0.25.1` (0.25.0 had corrupt protobuf files) | Check with `uv run python -c "import wandb"` if issues arise |
| `numpy` | `>=1.26.0,<2.0` | numpy 2.0 breaks several audio libs | Core dep |
| `ruff` | `>=0.9.0` | Linter; in `dev` extras, not core | Must be installed explicitly for linting |
| `flash-attn` | `>=2.5.0` | In eval extras; requires `torch` at build time | Falls back to PyTorch SDPA if unavailable |

### 1.3 Rules for adding new dependencies

1. **Check if it already exists** -- search `pyproject.toml` before adding.
2. **Choose the right scope** -- core only if needed at import time by both
   `train/` and `eval/`.  Training-only deps go in `train`, eval-only in
   `eval`, linters/formatters in `dev`.
3. **Pin conservatively** -- use `>=X.Y` for stable libs, `==X.Y.Z` only
   when a specific version is required for compatibility (e.g. `torchcodec`).
4. **Add source overrides** if the package comes from a non-PyPI index (e.g.
   PyTorch CUDA wheels) -- add an entry in `[tool.uv.sources]`.
5. **Run `uv lock`** after any change to regenerate the lockfile.
6. **Test the import** -- `uv run python -c "import <package>"` to verify.
7. **Never use `soundfile` for new code** -- use `torchaudio` for WAV I/O
   and `torchcodec` (via HuggingFace `datasets` Audio feature) for decoding.

### 1.4 Audio I/O stack

```
Reading audio from HuggingFace datasets  →  torchcodec  (datasets Audio feature)
Reading WAV files from disk              →  torchaudio.load()
Writing WAV files to disk                →  torchaudio.save()
Resampling                               →  torchaudio.functional.resample()
```

`soundfile` is still in core deps for legacy compatibility but should not be
used in new code.  `librosa` is used only for MFCC computation in
`bootstrap_eval.py`.

---

## 2. Linting with Ruff

### 2.1 What is Ruff?

[Ruff](https://docs.astral.sh/ruff/) is a Python linter and formatter written
in Rust by [Astral](https://astral.sh/) (the `uv` team).  It replaces flake8,
isort, pycodestyle, pyflakes, pydocstyle, and pyupgrade in a single binary
that runs 10-100x faster.

### 2.2 Installation

```bash
uv sync --extra all           # recommended: installs ruff along with everything else
# or just the dev extra:
uv sync --extra dev
```

Ruff is declared in `[project.optional-dependencies] dev` so it is not pulled
in by default -- you must request the `dev` extra.

### 2.3 Configuration (in `pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "D", "UP"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

### 2.4 Enabled rule sets

| Code | Rule set | What it catches |
|------|----------|-----------------|
| `E` | pycodestyle errors | Syntax errors, indentation, whitespace |
| `F` | pyflakes | Unused imports, undefined names, redefined variables |
| `W` | pycodestyle warnings | Trailing whitespace, blank lines, line length |
| `I` | isort | Import ordering (stdlib, third-party, local) |
| `D` | pydocstyle | Missing or malformed docstrings |
| `UP` | pyupgrade | Deprecated syntax (e.g. `dict()` -> `{}`, old-style type hints) |

### 2.5 Common commands

```bash
uv run ruff check .               # Check all files (requires dev or all extra)
uv run ruff check --fix .          # Auto-fix safe violations
uv run ruff check eval/            # Check only eval/
uv run ruff format .               # Auto-format (not enforced in CI yet)
```

---

## 3. Documentation Rules Applied Across All Files

Every Python file in this repository follows these rules.  They were
systematically applied on 2026-03-21 and must be maintained going forward.

### 3.1 Module-level docstring (required on every `.py` file)

Must include:

1. **One-line summary** of what the module does.
2. **Extended description** of how it works.
3. **Performance section** (if the module was optimised) with:
   - Technique name (e.g. "Batched GPU inference", "ProcessPoolExecutor").
   - Before/after wall-clock times.
   - Hardware the measurement was taken on.
   - Date the optimisation was added.
4. **Pipeline position** (for `eval/` modules) -- which stage number in
   `run_all.py`, what it depends on, what depends on it.
5. **Usage example** -- a copy-pasteable `uv run` command.
6. **License line** -- `License: MIT`.

### 3.2 Function / class docstrings (Google style)

Every public function and class must have a docstring with:

- One-line summary.
- Extended description (if non-trivial).
- `Args:` section listing every parameter with type and description.
- `Returns:` section describing the return value.
- `Raises:` section listing exceptions (if any are explicitly raised).

Private helper functions (`_foo()`) should have at least a one-line docstring.
Process-pool worker functions must document that they are workers, why they
accept a tuple, and what they return.

### 3.3 Inline comments

Required for:

- **Non-obvious math** (e.g. MCD formula, SNR dB conversion, Newton-Schulz).
- **Performance decisions** (e.g. `chunksize=32`, `batch_size=32`, why sort
  by length before batching, why cap workers at 16).
- **Compatibility workarounds** (e.g. protobuf shim, wandb version issues).
- **Module-level constants** (explain what they control and valid ranges).
- **Tricky control flow** (e.g. discriminator warmup, loss balancer
  `retain_graph`, Schedule-Free `optimizer.eval()` toggle).

### 3.4 Config YAML files

Every field must have an inline `#` comment explaining:
- What the field controls.
- Valid range or options.
- Why the current value was chosen (e.g. "Sweep-best LR from ooz250pm").

See `configs/base.yaml` as the reference style.

### 3.5 Shell scripts

Must have a header block explaining:
- Purpose.
- Invocation modes (if multiple, e.g. standalone vs. sharded VERSA).
- Environment variables consumed.
- Prerequisites.

---

## 4. Architecture Decisions

### 4.1 Evaluation pipeline (eval/run_all.py)

```
Stage 1: Reconstruction  (GPU, batched, bs=32)
    |
    +--> Stage 2: SSNR  (CPU, ProcessPoolExecutor, 16 workers) --+
    |                                                              |- concurrent
    +--> Stage 3: TTFAT  (GPU, micro-benchmark, 50 runs)  --------+
    |
Stage 4: Bootstrap       (CPU, ProcessPoolExecutor, 16 workers)
    |
Stage 5: VERSA           (sharded, 4 parallel bash subprocesses)
    |
Stage 6: WandB Publish   (single process, network-bound)
```

**Key design rules:**

- Stages 2+3 run concurrently because SSNR is CPU-only and TTFAT is GPU-only.
- Process pool workers must be top-level functions (pickle requirement).
- Worker functions accept a single tuple arg (Executor.map constraint).
- `chunksize` is tuned per stage: 32 for lightweight SSNR, 8 for heavy
  bootstrap (4 metrics per utterance including resampling).
- VERSA is an external tool invoked via `run_versa.sh`.  Parallelism is
  achieved by splitting the manifest into N scp shards and launching N
  subprocesses, each receiving shard paths via environment variables.
- State management: `EvalState` persists to JSON after every stage.
  Fingerprint hash of `(experiment, checkpoint, split, use_ema)` invalidates
  stale state.  Failed stages are retried; completed stages are skipped.

### 4.2 Codec registry (eval/codec_registry.py)

Adding a new codec requires only:
1. A `load` function: `(config, checkpoint, use_ema, device) -> model`.
2. An `encode_decode` function: `(model, waveform) -> reconstructed`.
3. (Optional) `extra_metrics`: list of `(name, fn)` for codec-specific stages.

No changes needed to `reconstruct.py`, `run_all.py`, `measure_ssnr.py`,
`bootstrap_eval.py`, `measure_ttfat.py`, `run_versa.sh`, or `log_to_wandb.py`.

### 4.3 Config inheritance (_bases_ merging)

```yaml
_bases_:
  - "../base.yaml"           # Applied first (defaults)
  - "../codecs/mimi.yaml"    # Overlays codec-specific fields
  - "../datasets/hindi.yaml" # Overlays dataset-specific fields
# Top-level keys here override everything above.
```

`train/config_loader.py` resolves bases recursively with deep-merge semantics.
Lists are replaced (not appended).  This is intentional -- a codec config
should be able to override the entire `losses` dict without inheriting stale
keys from `base.yaml`.

### 4.4 Optimizer factory (train/optimizer_factory.py)

Single function `create_optimizer()` returns `(optimizer, metadata_dict)`.
The metadata dict contains flags like `is_lr_free`, `is_schedule_free`,
`needs_three_betas` that the training loop and sweep agent use to adjust
behavior (e.g. skip LR scheduling for Prodigy, call `optimizer.eval()` for
Schedule-Free).

### 4.5 Audio I/O policy

- **torchcodec** for decoding from HuggingFace datasets (via `Audio` feature).
- **torchaudio** for all WAV read/write and resampling.
- **soundfile** is a legacy dependency -- do not use in new code.
- **librosa** is used only for MFCC in `bootstrap_eval.py`.

---

## 5. Performance Optimisation Guidelines

When adding or modifying evaluation stages, follow these patterns:

### 5.1 GPU-bound stages

- **Batch utterances** by sorting by length, padding per batch, and running a
  single model forward pass.  Default `batch_size=32`.
- Use `torch.no_grad()` and `torch.autocast(dtype=torch.bfloat16)`.
- Move data to GPU in bulk (`padded.to(device)`) rather than per-utterance.

### 5.2 CPU-bound stages (embarrassingly parallel)

- Use `concurrent.futures.ProcessPoolExecutor`.
- Cap `max_workers` at `min(os.cpu_count(), 16)` to avoid over-subscribing.
- Worker function must be a **top-level module function** (not a lambda,
  closure, or method) for `pickle` compatibility.
- Worker takes a **single tuple argument** (Executor.map constraint).
- Tune `chunksize`: larger (32-64) for lightweight ops (SSNR), smaller (4-8)
  for heavy ops (bootstrap with PESQ resampling).

### 5.3 External tool stages (VERSA)

- Split the input manifest into N shards.
- Write per-shard input files to a temp directory.
- Launch N parallel subprocesses via `ThreadPoolExecutor`.
- Pass shard paths via environment variables.
- Merge output files after all shards complete.

### 5.4 Stage-level concurrency

- If two stages use different resources (CPU vs GPU), run them concurrently
  via `ThreadPoolExecutor(max_workers=2)`.
- Always check the cache (`state.is_done()`) before launching.

### 5.5 Measuring and reporting improvements

When optimising a stage, record:
- **Before** wall-clock time (on a specific dataset size + hardware).
- **After** wall-clock time.
- **Technique** used.
- **Date** added.
- Add this to the module docstring under a `Performance` section.

---

## 6. Brainstorming: Future Improvement Opportunities

These are areas where the codebase could benefit from further work.  They are
not bugs -- they are ideas for contributors looking for impactful projects.

### 6.1 Evaluation pipeline

- **Streaming reconstruction**: Currently all waveforms are pre-loaded into
  CPU memory before batching.  For very large test sets (>10k utterances) this
  could hit memory limits.  A streaming DataLoader-based approach would fix
  this while preserving batching.
- **GPU-accelerated PESQ/STOI**: The bootstrap stage is CPU-bound because
  PESQ and STOI are numpy implementations.  GPU-native implementations
  (e.g. `torchmetrics` or custom CUDA kernels) could eliminate the process
  pool overhead entirely.
- **VERSA internal parallelism**: The current sharding approach launches
  independent processes.  If VERSA adds native batch/parallel support, the
  sharding wrapper could be simplified.
- **Adaptive batch size**: Instead of a fixed `batch_size=32`, dynamically
  choose based on available GPU memory and utterance lengths to maximise
  throughput without OOM.

### 6.2 Training

- **Multi-GPU training**: Currently single-GPU only.  `accelerate` is a
  declared dependency but not yet wired into `train_mimi.py`.
- **Gradient checkpointing**: For large codecs (DualCodec 200M + w2v-bert)
  on A100 40 GB, activation checkpointing would reduce memory pressure.
- **Mixed-precision discriminator**: The discriminator could run in fp32 while
  the generator uses bf16, improving GAN stability.

### 6.3 Data

- **On-the-fly resampling**: `prepare_data.py` resamples all audio upfront.
  For very large datasets, resampling on-the-fly in the DataLoader would save
  disk space.
- **Streaming HuggingFace datasets**: Currently the entire dataset is
  downloaded before preparation.  HuggingFace streaming mode could allow
  preparation with bounded memory.

### 6.4 Infrastructure

- **CI/CD pipeline**: No GitHub Actions yet.  A CI that runs `ruff check`,
  unit tests, and a smoke test on the Turkish sample would catch regressions.
- **Pre-commit hooks**: Ruff and type-checking via `mypy` or `pyright` as
  pre-commit hooks would enforce standards before code reaches the repo.
- **Benchmark regression tests**: Automated comparison of eval metrics against
  a known baseline to detect quality regressions from code changes.

---

## 7. Quick Reference: Common Tasks

| Task | Command |
|------|---------|
| Install everything | `uv sync --extra all` |
| Lint | `uv run ruff check .` |
| Auto-fix lint | `uv run ruff check --fix .` |
| Run tests | `uv run python -m unittest tests/test_repo_standards.py -v` |
| Syntax-check all Python files | `uv run python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p)]"` |
| Prepare Lahaja eval data | `uv run --extra all python scripts/prepare_lahaja.py` |
| Run full eval (Mimi Hindi) | `uv run python eval/run_all.py --config configs/experiments/mimi_hindi.yaml --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt --use-ema` |
| Run eval without VERSA/WandB | `uv run python eval/run_all.py --config configs/experiments/mimi_lahaja.yaml --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt --use-ema --skip-versa --skip-wandb` |
| Update lockfile after dep change | `uv lock` |
| Check wandb imports | `uv run python -c "import wandb; print(wandb.__version__)"` |
