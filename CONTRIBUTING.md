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

1. **Copy the template config:**
   ```bash
   cp configs/datasets/turkish_sample.yaml configs/datasets/your_dataset.yaml
   ```

2. **Fill in required fields:**
   - `name` — unique identifier for the dataset
   - `language` — ISO 639-1 code (e.g. `hi`, `tr`)
   - `source` — `local` or `huggingface`
   - `column_mappings` — map source columns to expected schema
   - `splits` — train / validation / test definitions
   - `filters` — any filtering criteria (duration, quality, etc.)

3. **Create experiment configs** for each codec:
   ```
   configs/experiments/{codec}_{dataset}.yaml
   ```

4. **Test data preparation:**
   ```bash
   uv run python prepare_data.py --config configs/datasets/your_dataset.yaml
   ```

---

## Adding a New Codec

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

3. **Wire up evaluation** — add a codec branch to `eval/reconstruct.py` that
   implements the encode/decode API for your codec.

4. **Wire up sweeps** — add a codec branch to `train/sweep_agent.py`.

5. **Create experiment and sweep configs:**
   ```
   configs/experiments/{your_codec}_{dataset}.yaml
   configs/sweeps/{your_codec}_sweep.yaml
   ```

6. **Test on the Turkish sample first** (3.1 h, fast iteration) before scaling
   to larger datasets like Hindi.

---

## Adding a New Optimizer

1. **Register the optimizer** in `train/optimizer_factory.py`:
   - Import the optimizer class
   - Add a creation function
   - Define parameter-group logic (e.g. separate LR for encoder vs. decoder)

2. **Document it** in `OPTIMIZERS.md`:
   - Add a row to the comparison table
   - Note sweep considerations and memory footprint

3. **Enable it in sweep configs** — add the optimizer name to the
   `optimizer.values` list in the relevant sweep YAML files.

4. **Handle special behavior** in `train/sweep_agent.py` if needed (e.g.
   Prodigy requires disabling the LR scheduler).

5. **Add the dependency** to `pyproject.toml` under
   `[project.optional-dependencies.train]`.

---

## Adding New Evaluation Metrics

1. Implement the metric computation in `eval/bootstrap_eval.py`.
2. Update `eval/log_to_wandb.py` to report the new metric.

---

## PR Checklist

Before submitting a pull request, make sure **all** of the following are true:

- [ ] `uv run ruff check .` passes
- [ ] All new functions have Google-style docstrings with type hints
- [ ] New config files have inline comments on every field
- [ ] Tested on the Turkish sample dataset (3.1 h, fast iteration)
- [ ] Updated `README.md` if adding user-facing features
- [ ] No secrets, API keys, or HuggingFace tokens in committed files
