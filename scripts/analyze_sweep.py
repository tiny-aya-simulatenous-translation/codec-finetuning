#!/usr/bin/env python3
"""Analyze a completed WandB sweep and export the best configuration.

Fetches all completed runs from a WandB sweep, ranks them by validation
loss, prints a summary table with parameter importance analysis, and
exports the best configuration as a YAML file.

Workflow
--------
1. **Fetch** -- Download all finished runs from the WandB API.
2. **Rank** -- Sort by ``val_loss`` ascending; display the top-K.
3. **Importance** -- Compute absolute Spearman rank correlation between
   each numeric hyperparameter and ``val_loss`` to highlight which knobs
   matter most.
4. **Export** -- Optionally merge the best run's hyperparameters into a
   base experiment YAML so the result can be used directly with
   ``train/train_mimi.py``.
5. **Log** -- Create a ``sweep-analysis`` run in WandB with the summary.

Usage::

    uv run python scripts/analyze_sweep.py \\
        --sweep-id <entity/project/sweep_id> \\
        --output configs/experiments/mimi_turkish_sample_best.yaml \\
        [--top-k 5]

Prerequisites:
    - Environment set up via ``scripts/setup.sh``
    - WandB login (``wandb login``)
    - A completed or partially completed sweep

License: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import wandb
import yaml


def fetch_sweep_runs(sweep_id: str) -> tuple[wandb.apis.public.Sweep, list[wandb.apis.public.Run]]:
    """Fetch all runs from a WandB sweep.

    Args:
        sweep_id: Full sweep identifier in the form ``entity/project/sweep_id``.

    Returns:
        A tuple of the sweep object and a list of finished runs sorted by
        ``val_loss`` ascending.
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    all_runs = sweep.runs

    finished_runs = [r for r in all_runs if r.state == "finished"]
    finished_runs.sort(key=lambda r: r.summary.get("val_loss", float("inf")))

    return sweep, finished_runs


def print_summary_table(
    sweep: wandb.apis.public.Sweep,
    finished_runs: list[wandb.apis.public.Run],
    all_run_count: int,
    top_k: int,
) -> None:
    """Print a formatted summary table of the top-K runs.

    Args:
        sweep: The WandB sweep object.
        finished_runs: Finished runs sorted by val_loss ascending.
        all_run_count: Total number of runs (including non-finished).
        top_k: Number of top runs to display.
    """
    killed = all_run_count - len(finished_runs)
    sweep_name = sweep.config.get("name", sweep.id)

    print("═══════════════════════════════════════════════════════════")
    print(f"Sweep Analysis: {sweep_name}")
    print(f"Total runs: {all_run_count} ({len(finished_runs)} finished, {killed} killed by Hyperband)")
    print("═══════════════════════════════════════════════════════════")

    header = f"{'Rank':<6}{'Run ID':<12}{'Optimizer':<18}{'LR':<12}{'WD':<10}{'Scheduler':<12}{'Val Loss'}"
    print(header)
    print("─" * len(header))

    for rank, run in enumerate(finished_runs[:top_k], start=1):
        config = run.config
        optimizer = config.get("optimizer", "unknown")
        lr = config.get("learning_rate", config.get("lr", "N/A"))
        wd = config.get("weight_decay", config.get("wd", "N/A"))
        scheduler = config.get("scheduler", "N/A")
        val_loss = run.summary.get("val_loss", float("inf"))

        # Format LR nicely
        if isinstance(lr, float):
            lr_str = f"{lr:.2e}" if lr < 0.001 else f"{lr:.4f}"
        else:
            lr_str = str(lr)

        # Format WD nicely
        if isinstance(wd, float):
            wd_str = f"{wd:.4f}"
        else:
            wd_str = str(wd)

        # Mark internal schedulers
        if optimizer in ("schedulefree", "prodigy") and scheduler == "N/A":
            scheduler = "(internal)"

        print(f"{rank:<6}{run.id:<12}{optimizer:<18}{lr_str:<12}{wd_str:<10}{scheduler:<12}{val_loss:.4f}")

    print("═══════════════════════════════════════════════════════════")


def compute_parameter_importance(
    finished_runs: list[wandb.apis.public.Run],
) -> dict[str, float]:
    """Compute parameter importance via correlation with val_loss.

    Uses the absolute Spearman rank correlation between each hyperparameter
    and the validation loss across all finished runs.

    Args:
        finished_runs: Finished runs sorted by val_loss ascending.

    Returns:
        A dictionary mapping parameter names to their absolute correlation
        with val_loss, sorted by importance descending.
    """
    # Need at least 3 runs for a meaningful rank correlation.
    if len(finished_runs) < 3:
        return {}

    val_losses = np.array([r.summary.get("val_loss", float("inf")) for r in finished_runs])

    # Collect all numeric hyperparameters across runs, skipping WandB
    # internal keys (prefixed with ``_``) and the version marker.
    param_values: dict[str, list[float]] = {}
    for run in finished_runs:
        for key, value in run.config.items():
            if key.startswith("_") or key in ("wandb_version",):
                continue
            if isinstance(value, (int, float)):
                param_values.setdefault(key, []).append(float(value))

    importance: dict[str, float] = {}
    for param_name, values in param_values.items():
        # Skip parameters that weren't logged for every run.
        if len(values) != len(finished_runs):
            continue
        arr = np.array(values)
        # Skip constant parameters -- zero variance means correlation
        # is undefined.
        if np.std(arr) < 1e-12:
            continue

        # Spearman rank correlation is robust to non-linear monotonic
        # relationships and outliers in val_loss.
        from scipy.stats import spearmanr

        corr, _ = spearmanr(arr, val_losses)
        importance[param_name] = abs(float(corr))

    # Sort by importance descending so the most influential knobs
    # appear first.
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def print_parameter_importance(importance: dict[str, float]) -> None:
    """Print a parameter importance ranking.

    Args:
        importance: Dictionary mapping parameter names to importance scores.
    """
    if not importance:
        print("\nParameter importance: Not enough runs for analysis.")
        return

    print("\nParameter Importance (|Spearman correlation| with val_loss):")
    print("─" * 50)
    for param, score in importance.items():
        bar = "█" * int(score * 30)
        print(f"  {param:<25} {score:.3f}  {bar}")
    print()


def export_best_config(
    best_run: wandb.apis.public.Run,
    base_config_path: Path | None,
    output_path: Path,
) -> None:
    """Export the best run's hyperparameters as a YAML config file.

    If a base config path is provided, the base config is loaded and the
    winning hyperparameters are merged on top. Otherwise, only the sweep
    hyperparameters are exported.

    Args:
        best_run: The best WandB run.
        base_config_path: Optional path to a base experiment config YAML.
        output_path: Path to write the output YAML.
    """
    # Load the base experiment config so we preserve all non-sweep
    # settings (codec, dataset, output_dir, etc.).
    if base_config_path and base_config_path.exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Strip WandB internal keys from the sweep hyperparameters.
    sweep_params = {
        k: v
        for k, v in best_run.config.items()
        if not k.startswith("_") and k != "wandb_version"
    }

    # Nest sweep params under "training" when the base config uses that
    # structure; otherwise place them at the top level.
    if "training" in config:
        config["training"].update(sweep_params)
    else:
        config.update(sweep_params)

    # Attach provenance metadata so the exported config can be traced
    # back to the sweep run that produced it.
    config.setdefault("_sweep_metadata", {})
    config["_sweep_metadata"]["source_run_id"] = best_run.id
    config["_sweep_metadata"]["val_loss"] = best_run.summary.get("val_loss")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Best config exported to: {output_path}")


def log_sweep_summary(
    sweep_id: str,
    finished_runs: list[wandb.apis.public.Run],
    importance: dict[str, float],
) -> None:
    """Log a summary of the sweep analysis to WandB.

    Creates a final summary run in the same project as the sweep to record
    the analysis results.

    Args:
        sweep_id: Full sweep identifier.
        finished_runs: Finished runs sorted by val_loss ascending.
        importance: Parameter importance dictionary.
    """
    parts = sweep_id.split("/")
    if len(parts) == 3:
        entity, project, _ = parts
    else:
        entity = None
        project = None

    best = finished_runs[0] if finished_runs else None
    summary_data: dict[str, Any] = {
        "total_finished_runs": len(finished_runs),
        "best_val_loss": best.summary.get("val_loss") if best else None,
        "best_run_id": best.id if best else None,
        "parameter_importance": importance,
    }

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="sweep-analysis",
        name=f"sweep-analysis-{sweep_id.split('/')[-1]}",
        config=summary_data,
    )
    if run is not None:
        run.summary.update(summary_data)
        run.finish()
        print("Sweep summary logged to WandB.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze a completed WandB sweep and export the best config.",
    )
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="Full WandB sweep ID (entity/project/sweep_id).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the best config YAML.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Base experiment config to merge winning params into.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top runs to display (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for sweep analysis.

    Parses CLI arguments, fetches sweep runs from WandB, prints the
    ranked summary table and parameter importance, optionally exports
    the best config YAML, and logs a sweep-analysis summary run.
    """
    args = parse_args()

    # Fetch runs
    print(f"Fetching sweep: {args.sweep_id}...")
    sweep, finished_runs = fetch_sweep_runs(args.sweep_id)
    all_run_count = len(sweep.runs)

    if not finished_runs:
        print("ERROR: No finished runs found in sweep.")
        sys.exit(1)

    # Print summary table
    print_summary_table(sweep, finished_runs, all_run_count, args.top_k)

    # Parameter importance
    importance = compute_parameter_importance(finished_runs)
    print_parameter_importance(importance)

    # Export best config
    if args.output:
        export_best_config(finished_runs[0], args.base_config, args.output)

    # Log summary to WandB
    log_sweep_summary(args.sweep_id, finished_runs, importance)


if __name__ == "__main__":
    main()
