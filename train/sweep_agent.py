"""WandB sweep agent for codec-finetuning hyperparameter search.

This script is the entry point for WandB sweep runs.  It receives
hyperparameters from the WandB sweep controller, applies them to the
base experiment config, and runs a single training job.

Sweep flow
----------
1. ``wandb agent`` spawns this script as a subprocess.
2. :func:`main` calls ``wandb.init()`` which populates
   ``wandb.config`` with the trial's hyperparameter values.
3. The ``codec`` and ``dataset_config`` keys are popped to locate the
   correct base experiment YAML.
4. :func:`~train.config_loader.apply_sweep_overrides` maps the
   remaining flat sweep parameters onto the nested config dict.
5. The resolved config is validated and dispatched to the appropriate
   codec training function (Mimi uses a direct Python call; DualCodec
   and Kanade use subprocess-based launchers).

Supported codecs
----------------
- **mimi** — calls :func:`train.train_mimi.train` in-process.
- **dualcodec** — delegates to ``train/train_dualcodec.sh`` via
  ``subprocess``.
- **kanade** — delegates to ``train/train_kanade.sh`` via
  ``subprocess``.

Usage::

    # Do NOT run this script directly.  Use the sweep helper:
    bash scripts/run_sweep.sh mimi

    # Or manually:
    wandb sweep configs/sweeps/mimi_sweep.yaml
    wandb agent <sweep_id>

License: MIT
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import wandb
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.config_loader import apply_sweep_overrides, load_config, validate_config


def _write_temp_config(config: dict) -> Path:
    """Write a resolved config to a secure temporary YAML file.

    Args:
        config: Fully resolved experiment configuration.

    Returns:
        Path to the temporary YAML file.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)
        return Path(fh.name)


def main() -> None:
    """Run a single sweep trial with WandB-provided hyperparameters.

    Initialises a WandB run, retrieves sweep parameters from the sweep
    controller, loads the corresponding base experiment config, applies
    the parameter overrides, and dispatches to the appropriate codec
    training pipeline.

    Raises:
        ValueError: If the ``codec`` parameter is not one of the supported
            values (``mimi``, ``dualcodec``, ``kanade``).
    """
    parser = argparse.ArgumentParser(
        description="WandB sweep agent entry point for codec fine-tuning.",
    )
    parser.add_argument(
        "sweep_params_file",
        nargs="?",
        help="JSON file with sweep params (provided by wandb agent)",
    )
    parser.parse_args()

    # ── Step 1: WandB initialisation ────────────────────────────────────
    # wandb.init() contacts the sweep controller and populates
    # wandb.config with the trial's hyperparameter sample.
    run = wandb.init()
    sweep_params: dict = dict(wandb.config)

    # ── Step 2: Determine codec and dataset ──────────────────────────────
    # These two meta-keys are consumed here and removed before the
    # remaining sweep_params are forwarded to apply_sweep_overrides.
    codec: str = sweep_params.pop("codec", "mimi")
    dataset_config: str = sweep_params.pop(
        "dataset_config", "configs/datasets/turkish_sample.yaml"
    )

    # ── Step 3: Load the base experiment config ──────────────────────────
    # Convention: experiment configs live at
    # configs/experiments/{codec}_{dataset_name}.yaml
    dataset_name: str = Path(dataset_config).stem  # e.g., "turkish_sample"
    experiment_config_path: str = f"configs/experiments/{codec}_{dataset_name}.yaml"

    config: dict = load_config(experiment_config_path)

    # ── Step 4: Apply sweep hyperparameter overrides ─────────────────────
    config = apply_sweep_overrides(config, sweep_params)

    # Tag the WandB run for easy filtering in the dashboard.
    config["wandb"]["tags"] = [codec, dataset_name, "sweep"]
    config["wandb"]["run_name"] = f"{codec}-sweep-{run.id}"

    # Isolate sweep outputs so they don't overwrite manual runs.
    config["output_dir"] = f"outputs/sweeps/{codec}_{dataset_name}/{run.id}"

    # ── Step 5: Validate and log resolved config ─────────────────────────
    validate_config(config)
    wandb.config.update({"resolved_config": config}, allow_val_change=True)

    # ── Step 6: Dispatch to the correct training pipeline ────────────────
    if codec == "mimi":
        from train.train_mimi import train

        train(config)
    elif codec == "dualcodec":
        _run_dualcodec(config)
    elif codec == "kanade":
        _run_kanade(config)
    else:
        raise ValueError(
            f"Unknown codec: {codec}. Supported: mimi, dualcodec, kanade"
        )

    wandb.finish()


def _run_dualcodec(config: dict) -> None:
    """Launch DualCodec training via its native accelerate+Hydra pipeline.

    Serialises the resolved config to a temporary YAML file and invokes
    the ``train/train_dualcodec.sh`` launcher script through ``uv run``.

    Args:
        config: Fully resolved experiment configuration dictionary.
    """
    config_path = _write_temp_config(config)

    try:
        cmd = [
            "uv",
            "run",
            "bash",
            "train/train_dualcodec.sh",
            "--config",
            str(config_path),
        ]
        subprocess.run(cmd, check=True, capture_output=False)
    finally:
        config_path.unlink(missing_ok=True)


def _run_kanade(config: dict) -> None:
    """Launch Kanade training via its native Lightning CLI pipeline.

    Serialises the resolved config to a temporary YAML file and invokes
    the ``train/train_kanade.sh`` launcher script through ``uv run``.

    Args:
        config: Fully resolved experiment configuration dictionary.
    """
    config_path = _write_temp_config(config)

    try:
        cmd = [
            "uv",
            "run",
            "bash",
            "train/train_kanade.sh",
            "--config",
            str(config_path),
        ]
        subprocess.run(cmd, check=True, capture_output=False)
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
