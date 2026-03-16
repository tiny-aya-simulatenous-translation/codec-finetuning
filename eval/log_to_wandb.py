"""Log evaluation results to Weights & Biases (WandB).

Aggregates metrics from bootstrap evaluation, TTFAT, and SSNR JSON files
and logs them as WandB summary metrics, a per-utterance table, and audio
samples.

Usage::

    uv run python eval/log_to_wandb.py \
        --experiment mimi_turkish_sample \
        [--run-id <wandb_run_id>]

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from train.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning ``None`` if it does not exist.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict or ``None`` if file is missing.
    """
    if not path.exists():
        logger.warning("JSON not found, skipping: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pick_audio_samples(
    manifest: List[Dict[str, Any]],
    n: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select N random entries from a manifest for audio logging.

    Args:
        manifest: List of manifest entries.
        n: Number of samples to pick.
        seed: Random seed for reproducibility.

    Returns:
        List of selected manifest entries.
    """
    rng = random.Random(seed)
    n = min(n, len(manifest))
    return rng.sample(manifest, n)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def log_to_wandb(
    experiment: str,
    run_id: Optional[str] = None,
) -> None:
    """Log all evaluation metrics to WandB.

    Args:
        experiment: Experiment name (maps to a config under
            ``configs/experiments/``).
        run_id: Optional WandB run ID to resume. Creates a new run if
            ``None``.

    Raises:
        FileNotFoundError: If the experiment config is missing.
    """
    import wandb

    config_path = Path("configs/experiments") / f"{experiment}.yaml"
    config = load_config(str(config_path))
    sample_rate = int(config["codec"]["sample_rate"])

    # Load all metric JSON files.
    results_dir = Path("results")
    bootstrap_data = _load_json(results_dir / f"{experiment}_metrics.json")
    ttfat_data = _load_json(results_dir / f"{experiment}_ttfat.json")
    ssnr_data = _load_json(results_dir / f"{experiment}_ssnr.json")

    # Initialise or resume WandB run.
    wandb_kwargs: Dict[str, Any] = {
        "project": config.get("wandb", {}).get("project", "codec-finetuning"),
        "config": config,
        "tags": config.get("wandb", {}).get("tags", []) + ["eval"],
        "job_type": "eval",
    }
    if run_id is not None:
        wandb_kwargs["id"] = run_id
        wandb_kwargs["resume"] = "must"
    else:
        wandb_kwargs["name"] = f"{experiment}-eval"

    run = wandb.init(**wandb_kwargs)

    # ── Log bootstrap metrics as summary ─────────────────────────────
    if bootstrap_data is not None:
        metrics = bootstrap_data.get("metrics", {})
        for metric_name, values in metrics.items():
            run.summary[f"eval/{metric_name}_mean"] = values["mean"]
            run.summary[f"eval/{metric_name}_std"] = values["std"]
            run.summary[f"eval/{metric_name}_ci_low"] = values["ci_low"]
            run.summary[f"eval/{metric_name}_ci_high"] = values["ci_high"]
        logger.info("Logged bootstrap metrics to WandB summary.")

    # ── Log TTFAT metrics ────────────────────────────────────────────
    if ttfat_data is not None:
        run.summary["eval/ttfat_mean_ms"] = ttfat_data["mean_ms"]
        run.summary["eval/ttfat_std_ms"] = ttfat_data["std_ms"]
        run.summary["eval/ttfat_p50_ms"] = ttfat_data["p50_ms"]
        run.summary["eval/ttfat_p95_ms"] = ttfat_data["p95_ms"]
        run.summary["eval/ttfat_p99_ms"] = ttfat_data["p99_ms"]
        logger.info("Logged TTFAT metrics to WandB summary.")

    # ── Log SSNR metrics ─────────────────────────────────────────────
    if ssnr_data is not None:
        run.summary["eval/ssnr_mean"] = ssnr_data["ssnr_mean"]
        run.summary["eval/ssnr_std"] = ssnr_data["ssnr_std"]
        logger.info("Logged SSNR metrics to WandB summary.")

    # ── Log per-utterance table ──────────────────────────────────────
    recon_manifest_path = (
        Path(config["output_dir"]) / "reconstructed" / "test" / "manifest.json"
    )
    recon_manifest: Optional[List[Dict[str, Any]]] = None
    if recon_manifest_path.exists():
        with open(recon_manifest_path, "r", encoding="utf-8") as fh:
            recon_manifest = json.load(fh)

    if recon_manifest is not None and bootstrap_data is not None:
        columns = ["id", "duration"] + sorted(
            bootstrap_data.get("metrics", {}).keys()
        )
        table = wandb.Table(columns=columns)

        # Per-utterance metrics would need to be recomputed; log manifest
        # entries with available metadata.
        for entry in recon_manifest:
            row = [entry.get("id", ""), entry.get("duration", 0.0)]
            # Fill metric columns with placeholders (full per-utterance
            # values would require re-running compute_utterance_metrics).
            row.extend([None] * (len(columns) - 2))
            table.add_data(*row)

        run.log({"eval/per_utterance": table})
        logger.info("Logged per-utterance table (%d rows).", len(recon_manifest))

    # ── Log audio samples ────────────────────────────────────────────
    if recon_manifest is not None:
        samples = _pick_audio_samples(recon_manifest, n=5)
        for entry in samples:
            utt_id = entry.get("id", "unknown")
            orig_path = entry.get("original_path")
            recon_path = entry.get("reconstructed_path")

            if orig_path and Path(orig_path).exists():
                run.log({
                    f"eval/audio/{utt_id}_original": wandb.Audio(
                        orig_path, sample_rate=sample_rate, caption="Original",
                    ),
                })
            if recon_path and Path(recon_path).exists():
                run.log({
                    f"eval/audio/{utt_id}_reconstructed": wandb.Audio(
                        recon_path, sample_rate=sample_rate,
                        caption="Reconstructed",
                    ),
                })

        logger.info("Logged %d audio sample pairs.", len(samples))

    wandb.finish()
    logger.info("WandB logging complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and log evaluation results to WandB."""
    parser = argparse.ArgumentParser(
        description="Log evaluation results to WandB.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g. mimi_turkish_sample).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="WandB run ID to resume. Creates a new run if omitted.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_to_wandb(
        experiment=args.experiment,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
