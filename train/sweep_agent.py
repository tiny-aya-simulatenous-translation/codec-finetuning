"""WandB sweep agent for codec-finetuning hyperparameter search.

This script is the entry point for WandB sweep runs. It receives hyperparameters
from the WandB sweep controller, applies them to the base experiment config,
and runs a single training job.

Usage:
    This script is called automatically by ``wandb agent``. Do not run directly.
    Instead, use::

        bash scripts/run_sweep.sh mimi

    Or manually::

        wandb sweep configs/sweeps/mimi_sweep.yaml
        wandb agent <sweep_id>

License: MIT
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import wandb
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.config_loader import apply_sweep_overrides, load_config, validate_config


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
    args = parser.parse_args()

    # Initialize WandB run (sweep controller provides the config)
    run = wandb.init()
    sweep_params: dict = dict(wandb.config)

    # Determine codec and load base config
    codec: str = sweep_params.pop("codec", "mimi")
    dataset_config: str = sweep_params.pop(
        "dataset_config", "configs/datasets/turkish_sample.yaml"
    )

    # Load the base experiment config for this codec + dataset.
    # Map codec name to experiment config.
    dataset_name: str = Path(dataset_config).stem  # e.g., "turkish_sample"
    experiment_config_path: str = f"configs/experiments/{codec}_{dataset_name}.yaml"

    config: dict = load_config(experiment_config_path)

    # Apply sweep overrides
    config = apply_sweep_overrides(config, sweep_params)

    # Update WandB tags
    config["wandb"]["tags"] = [codec, dataset_name, "sweep"]
    config["wandb"]["run_name"] = f"{codec}-sweep-{run.id}"

    # Update output dir to be sweep-specific
    config["output_dir"] = f"outputs/sweeps/{codec}_{dataset_name}/{run.id}"

    # Validate
    validate_config(config)

    # Log full config to WandB
    wandb.config.update({"resolved_config": config}, allow_val_change=True)

    # Import and run the appropriate training function
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
    config_path = Path(tempfile.mktemp(suffix=".yaml"))
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    cmd = [
        "uv",
        "run",
        "bash",
        "train/train_dualcodec.sh",
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, check=True, capture_output=False)


def _run_kanade(config: dict) -> None:
    """Launch Kanade training via its native Lightning CLI pipeline.

    Serialises the resolved config to a temporary YAML file and invokes
    the ``train/train_kanade.sh`` launcher script through ``uv run``.

    Args:
        config: Fully resolved experiment configuration dictionary.
    """
    config_path = Path(tempfile.mktemp(suffix=".yaml"))
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    cmd = [
        "uv",
        "run",
        "bash",
        "train/train_kanade.sh",
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, check=True, capture_output=False)


if __name__ == "__main__":
    main()
