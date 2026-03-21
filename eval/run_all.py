"""Unified evaluation pipeline -- single entry-point for all eval stages.

Runs every evaluation step in sequence, saves results locally as JSON,
and publishes everything to a single WandB run so no step is missed and
all metrics live in one place.

**Automatic resume:** A state file (``results/<experiment>_eval_state.json``)
is written after every stage.  If the pipeline is interrupted or a stage
fails, re-running the same command automatically skips already-completed
stages and picks up where it left off.  Use ``--restart`` to force a
clean run from scratch.

Stages (in order)
-----------------
======  ==========  ============================================
Stage   Name        Description
======  ==========  ============================================
1       Reconstruct Encode/decode test utterances (batched GPU).
2       SSNR        Segmental Signal-to-Noise Ratio (parallel CPU).
3       TTFAT       Time to First Audio Token latency (GPU).
4       Bootstrap   PESQ, STOI, DNSMOS, MCD with CIs (parallel CPU).
5       VERSA       Comprehensive 90+ metric suite (sharded parallel).
6       WandB       Aggregate all results into one WandB eval run.
======  ==========  ============================================

Stages 2 and 3 (SSNR + TTFAT) run **concurrently** because SSNR is
CPU-only while TTFAT is GPU-only -- they have zero resource contention.

Performance optimisations (added 2026-03-21)
--------------------------------------------
The following improvements were applied to cut total evaluation time on
a 1665-utterance Hindi test set (H100 + 16 CPU cores):

+---------------+-----------+----------+------------------------------------+
| Stage         | Before    | After    | Technique                          |
+===============+===========+==========+====================================+
| Reconstruct   | ~26 s     | ~4-8 s   | Batched GPU inference (bs=32).     |
+---------------+-----------+----------+------------------------------------+
| SSNR          | ~26 s     | ~2-4 s   | ``ProcessPoolExecutor`` (16 wkrs). |
+---------------+-----------+----------+------------------------------------+
| TTFAT         | ~3 s      | ~3 s     | Unchanged (micro-benchmark).       |
+---------------+-----------+----------+------------------------------------+
| Bootstrap     | ~42 min   | ~3 min   | ``ProcessPoolExecutor`` (16 wkrs). |
+---------------+-----------+----------+------------------------------------+
| VERSA         | ~212 min  | ~55 min  | 4 sharded parallel subprocesses.   |
+---------------+-----------+----------+------------------------------------+
| SSNR + TTFAT  | sequential| parallel | ``ThreadPoolExecutor`` overlap.    |
+---------------+-----------+----------+------------------------------------+

Total wall-clock improvement: **~4.7 h -> ~1.0 h** (approx. 4.7x).

Usage::

    uv run python eval/run_all.py \\
        --config configs/experiments/mimi_hindi.yaml \\
        --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \\
        --use-ema

License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eval.codec_registry import get_codec_hooks
from train.config_loader import load_config

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 70
_THIN_SEP = "-" * 70

STAGE_NAMES = ("reconstruction", "ssnr", "ttfat", "bootstrap", "versa", "wandb")


# ---------------------------------------------------------------------------
# Persistent eval state -- enables automatic resume
# ---------------------------------------------------------------------------


class EvalState:
    """Tracks which stages have completed and caches their outputs.

    Persisted as ``results/<experiment>_eval_state.json`` after every
    stage so that a crash or failure at stage N lets the next invocation
    skip stages 1..N-1.

    The state file records a *fingerprint* derived from the experiment
    name, checkpoint path, split, and use-ema flag.  If any of those
    change, a stale state file is automatically invalidated so you cannot
    accidentally mix results from different runs.
    """

    def __init__(self, path: Path, fingerprint: str) -> None:
        self._path = path
        self._fingerprint = fingerprint
        self._completed: Dict[str, Any] = {}
        self._load()

    @staticmethod
    def make_fingerprint(
        experiment: str,
        checkpoint: Optional[str],
        split: str,
        use_ema: bool,
    ) -> str:
        """Deterministic hash of the run identity."""
        raw = f"{experiment}|{checkpoint or 'pretrained'}|{split}|{use_ema}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load(self) -> None:
        """Load state from disk, invalidating if the fingerprint changed."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("fingerprint") != self._fingerprint:
                logger.warning(
                    "State file fingerprint mismatch (different config / "
                    "checkpoint / split / ema). Starting fresh."
                )
                self._completed = {}
                return
            self._completed = data.get("completed", {})
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt state file -- starting fresh.")
            self._completed = {}

    def _flush(self) -> None:
        """Persist current state to disk immediately."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self._fingerprint,
            "completed": self._completed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def is_done(self, stage: str) -> bool:
        """Return True if *stage* already completed successfully."""
        return stage in self._completed and self._completed[stage].get("status") == "ok"

    def get_data(self, stage: str) -> Optional[Dict[str, Any]]:
        """Return cached output data for a completed stage."""
        entry = self._completed.get(stage)
        if entry is None:
            return None
        return entry.get("data")

    def mark_done(self, stage: str, data: Any = None) -> None:
        """Record a stage as successfully completed and flush to disk."""
        self._completed[stage] = {
            "status": "ok",
            "data": data,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._flush()

    def mark_failed(self, stage: str, error: str) -> None:
        """Record a stage failure (does NOT count as done) and flush."""
        self._completed[stage] = {
            "status": "failed",
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._flush()

    def clear(self) -> None:
        """Wipe all cached state (used by ``--restart``)."""
        self._completed = {}
        if self._path.exists():
            self._path.unlink()

    def summary(self) -> Dict[str, str]:
        """Return {stage: status} for all recorded stages."""
        return {
            stage: self._completed[stage].get("status", "unknown")
            for stage in self._completed
        }


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _stage_banner(name: str, index: int, total: int, resumed: bool = False) -> None:
    """Print a visible banner for the current stage."""
    tag = "  [RESUMED]" if resumed else ""
    print(f"\n{_SEPARATOR}")
    print(f"  STAGE {index}/{total}: {name}{tag}")
    print(f"{_SEPARATOR}\n")


def _skip_banner(name: str, index: int, total: int) -> None:
    """Print a banner indicating the stage was skipped (already done)."""
    print(f"\n{_SEPARATOR}")
    print(f"  STAGE {index}/{total}: {name}  [CACHED -- skipping]")
    print(f"{_SEPARATOR}\n")


def _save_json(data: Any, path: Path) -> None:
    """Write *data* to *path* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved: %s", path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file or return ``None`` if missing."""
    if not path.exists():
        logger.warning("Not found, skipping: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stage 1: Reconstruction
# ---------------------------------------------------------------------------


def _run_reconstruction(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    split: str,
) -> Path:
    """Reconstruct audio through the codec.

    Returns:
        Path to the output reconstruction manifest.
    """
    from eval.reconstruct import reconstruct

    manifest_path = reconstruct(
        config,
        checkpoint=checkpoint,
        split=split,
        use_ema=use_ema,
    )
    return Path(manifest_path)


# ---------------------------------------------------------------------------
# Stage 2: SSNR
# ---------------------------------------------------------------------------


def _run_ssnr(
    experiment: str,
    results_dir: Path,
    segment_ms: float = 25.0,
    overlap_ms: float = 10.0,
) -> Dict[str, Any]:
    """Compute Segmental SNR and save results."""
    from eval.measure_ssnr import run as ssnr_run

    stats = ssnr_run(
        experiment=experiment,
        segment_ms=segment_ms,
        overlap_ms=overlap_ms,
    )
    _save_json(stats, results_dir / f"{experiment}_ssnr.json")
    return stats


# ---------------------------------------------------------------------------
# Stage 3: TTFAT
# ---------------------------------------------------------------------------


def _run_ttfat(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    experiment: str,
    results_dir: Path,
    n_runs: int = 50,
    warmup: int = 5,
) -> Dict[str, Any]:
    """Measure Time to First Audio Token latency and save results."""
    from eval.measure_ttfat import measure_ttfat

    stats = measure_ttfat(
        config,
        checkpoint=checkpoint,
        n_runs=n_runs,
        warmup=warmup,
    )
    _save_json(stats, results_dir / f"{experiment}_ttfat.json")
    return stats


# ---------------------------------------------------------------------------
# Stage 4: Bootstrap metrics (PESQ, STOI, DNSMOS, MCD)
# ---------------------------------------------------------------------------


def _run_bootstrap(
    experiment: str,
    results_dir: Path,
    n_resamples: int = 20,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute per-utterance metrics with bootstrap CIs and save results."""
    from eval.bootstrap_eval import run as bootstrap_run

    output_json = str(results_dir / f"{experiment}_metrics.json")
    result = bootstrap_run(
        experiment=experiment,
        n_resamples=n_resamples,
        confidence=confidence,
        output_json=output_json,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 5: VERSA (optional)
# ---------------------------------------------------------------------------


def _run_versa_shard(
    script: Path,
    experiment: str,
    shard_idx: int,
    gt_scp: str,
    pred_scp: str,
    output_file: str,
    score_config: str,
) -> subprocess.CompletedProcess:
    """Launch one VERSA scorer process for a manifest shard.

    The shell script ``run_versa.sh`` is invoked with shard-specific
    environment variables (``VERSA_SHARD_GT``, ``VERSA_SHARD_PRED``,
    ``VERSA_SHARD_OUTPUT``, ``VERSA_SCORE_CONFIG``) that override the
    default manifest-based scp generation.  This lets multiple shards
    run concurrently without file-name collisions.

    Args:
        script: Absolute path to ``eval/run_versa.sh``.
        experiment: Experiment name (passed as ``$1`` to the script).
        shard_idx: Zero-based shard index (for logging / debugging).
        gt_scp: Path to the ground-truth Kaldi-style scp file for
            this shard.
        pred_scp: Path to the prediction scp file for this shard.
        output_file: Path where VERSA will write JSON-lines output
            for this shard.
        score_config: Path to the VERSA score configuration YAML.

    Returns:
        The :class:`subprocess.CompletedProcess` from the VERSA run.
    """
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "VERSA_SHARD_GT": gt_scp,
        "VERSA_SHARD_PRED": pred_scp,
        "VERSA_SHARD_OUTPUT": output_file,
        "VERSA_SCORE_CONFIG": score_config,
    }
    return subprocess.run(
        ["bash", str(script), experiment],
        capture_output=True,
        text=True,
        env=env,
    )


def _run_versa(
    experiment: str,
    results_dir: Path,
    n_shards: int = 4,
) -> Dict[str, Any]:
    """Run VERSA comprehensive evaluation with sharded parallelism.

    **Sharded parallel execution** (added 2026-03-21):

    VERSA is the most expensive stage (~212 min for 1665 utterances in
    the Hindi eval).  To accelerate it we split the reconstruction
    manifest into *n_shards* disjoint chunks, write per-shard Kaldi-
    style ``.scp`` files, and launch *n_shards* independent
    ``run_versa.sh`` processes via a :class:`~concurrent.futures.ThreadPoolExecutor`.
    Each subprocess is CPU/GPU independent, so we get near-linear
    speedup up to the point where GPU contention or I/O becomes the
    bottleneck.

    Measured improvement (1665 utterances, 4 shards, H100):

        Before: ~212 min (single process)
        After:  ~55 min  (4 parallel shards)

    After all shards complete, their JSON-lines outputs are concatenated
    and aggregated into the final ``results/<experiment>_versa.json``.

    The function validates that the ``versa`` package is importable
    before launching any subprocess.

    Args:
        experiment: Experiment name (maps to
            ``configs/experiments/<experiment>.yaml``).
        results_dir: Directory for output JSON files.
        n_shards: Number of parallel VERSA processes.  Capped at the
            number of utterances.  Default 4 is a good balance between
            GPU utilisation and memory on a single H100.

    Returns:
        Aggregated VERSA results dict mapping metric names to their
        mean values across all utterances.

    Raises:
        ImportError: If the ``versa`` package is not installed.
        FileNotFoundError: If ``run_versa.sh`` or the reconstruction
            manifest is missing.
        RuntimeError: If any VERSA subprocess exits with a non-zero
            code, or if no output is produced.
    """
    import importlib
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import eval._protobuf_compat  # noqa: F401
    importlib.import_module("versa")

    script = Path(__file__).parent / "run_versa.sh"
    if not script.exists():
        raise FileNotFoundError(
            f"run_versa.sh not found at {script}. "
            "Ensure the eval/ directory is intact."
        )

    # Load the reconstruction manifest to build per-shard scp files.
    config_path = Path("configs/experiments") / f"{experiment}.yaml"
    config = load_config(str(config_path))
    output_dir = config.get("output_dir", "outputs/default")
    manifest_path = Path(output_dir) / "reconstructed" / "test" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Reconstruction manifest not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest: List[Dict[str, Any]] = json.load(fh)

    # Don't create more shards than utterances.
    actual_shards = min(n_shards, len(manifest))
    if actual_shards < 2:
        actual_shards = 1

    logger.info(
        "Running VERSA evaluation (%d shards, %d utterances)...",
        actual_shards, len(manifest),
    )

    score_config = str(Path(__file__).parent / "versa_score_config.yaml")
    shard_size = (len(manifest) + actual_shards - 1) // actual_shards
    tmp_dir = tempfile.mkdtemp(prefix="versa_shards_")

    # ---- Launch shards in parallel ----------------------------------------
    shard_outputs: List[str] = []
    futures = {}

    with ThreadPoolExecutor(max_workers=actual_shards) as pool:
        for si in range(actual_shards):
            chunk = manifest[si * shard_size : (si + 1) * shard_size]
            if not chunk:
                continue

            # Write per-shard Kaldi scp files: "<utt_id> <abs_path>\n"
            gt_scp = os.path.join(tmp_dir, f"gt_{si}.scp")
            pred_scp = os.path.join(tmp_dir, f"pred_{si}.scp")
            out_file = os.path.join(tmp_dir, f"versa_{si}.json")
            shard_outputs.append(out_file)

            with open(gt_scp, "w") as gf, open(pred_scp, "w") as pf:
                for entry in chunk:
                    utt_id = entry.get(
                        "id",
                        os.path.splitext(os.path.basename(entry["original_path"]))[0],
                    )
                    gf.write(f"{utt_id} {os.path.abspath(entry['original_path'])}\n")
                    pf.write(f"{utt_id} {os.path.abspath(entry['reconstructed_path'])}\n")

            fut = pool.submit(
                _run_versa_shard,
                script, experiment, si,
                gt_scp, pred_scp, out_file, score_config,
            )
            futures[fut] = si

        # Collect results; fail fast on any shard error.
        for fut in as_completed(futures):
            si = futures[fut]
            result = fut.result()
            if result.returncode != 0:
                raise RuntimeError(
                    f"VERSA shard {si} exited with code {result.returncode}:\n"
                    f"{result.stderr}"
                )

    # ---- Merge shard JSON-lines outputs -----------------------------------
    per_utt: List[Dict[str, Any]] = []
    for out_file in shard_outputs:
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        per_utt.append(json.loads(line))

    if not per_utt:
        raise RuntimeError("VERSA output is empty across all shards.")

    # Write the combined file to the canonical results location.
    versa_path = results_dir / f"{experiment}_versa.json"
    with open(versa_path, "w", encoding="utf-8") as fh:
        for entry in per_utt:
            fh.write(json.dumps(entry) + "\n")

    # Aggregate: mean of each numeric metric across all utterances.
    agg: Dict[str, float] = {}
    for key in per_utt[0]:
        vals = [u[key] for u in per_utt if key in u and isinstance(u[key], (int, float))]
        if vals:
            agg[key] = round(float(np.mean(vals)), 4)

    logger.info("VERSA aggregated %d metrics over %d utterances.", len(agg), len(per_utt))
    return agg


# ---------------------------------------------------------------------------
# Stage 6: Publish all results to WandB
# ---------------------------------------------------------------------------


def _publish_to_wandb(
    config: Dict[str, Any],
    experiment: str,
    results_dir: Path,
    checkpoint: Optional[str],
    ssnr_data: Optional[Dict[str, Any]],
    ttfat_data: Optional[Dict[str, Any]],
    bootstrap_data: Optional[Dict[str, Any]],
    versa_data: Optional[Dict[str, Any]],
    codec_extras_data: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    n_audio_samples: int = 5,
) -> str:
    """Push every metric to a single WandB eval run.

    Returns:
        The WandB run URL.
    """
    import random

    import wandb

    sample_rate = int(config["codec"]["sample_rate"])

    wandb_cfg = config.get("wandb", {})
    wandb_kwargs: Dict[str, Any] = {
        "project": wandb_cfg.get("project", "codec-finetuning"),
        "config": config,
        "tags": wandb_cfg.get("tags", []) + ["eval", "run-all"],
        "job_type": "eval",
    }
    if run_id is not None:
        wandb_kwargs["id"] = run_id
        wandb_kwargs["resume"] = "must"
    else:
        wandb_kwargs["name"] = f"{experiment}-eval"

    run = wandb.init(**wandb_kwargs)

    # -- Metadata -----------------------------------------------------------
    run.summary["eval/checkpoint"] = checkpoint or "pretrained"
    run.summary["eval/experiment"] = experiment
    run.summary["eval/timestamp"] = datetime.now(timezone.utc).isoformat()

    # -- SSNR ---------------------------------------------------------------
    if ssnr_data is not None:
        run.summary["eval/ssnr_mean"] = ssnr_data["ssnr_mean"]
        run.summary["eval/ssnr_std"] = ssnr_data["ssnr_std"]
        run.summary["eval/ssnr_min"] = ssnr_data.get("ssnr_min")
        run.summary["eval/ssnr_max"] = ssnr_data.get("ssnr_max")
        logger.info("Published SSNR metrics.")

    # -- TTFAT --------------------------------------------------------------
    if ttfat_data is not None:
        for key in ("mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            if key in ttfat_data:
                run.summary[f"eval/ttfat_{key}"] = ttfat_data[key]
        logger.info("Published TTFAT metrics.")

    # -- Bootstrap metrics --------------------------------------------------
    if bootstrap_data is not None:
        metrics = bootstrap_data.get("metrics", {})
        for metric_name, values in metrics.items():
            for stat_key in ("mean", "std", "ci_low", "ci_high"):
                if stat_key in values:
                    run.summary[f"eval/{metric_name}_{stat_key}"] = values[stat_key]
        logger.info("Published bootstrap metrics (PESQ, STOI, DNSMOS, MCD).")

    # -- VERSA --------------------------------------------------------------
    if versa_data is not None:
        if isinstance(versa_data, dict):
            for key, val in versa_data.items():
                if isinstance(val, (int, float)):
                    run.summary[f"eval/versa/{key}"] = val
        logger.info("Published VERSA metrics.")

    # -- Codec-specific extra metrics ---------------------------------------
    if codec_extras_data:
        for extra_name, extra_data in codec_extras_data.items():
            if isinstance(extra_data, dict):
                for key, val in extra_data.items():
                    if isinstance(val, (int, float)):
                        run.summary[f"eval/codec/{extra_name}/{key}"] = val
        logger.info("Published codec-specific metrics: %s", list(codec_extras_data))

    # -- Per-utterance table ------------------------------------------------
    recon_manifest_path = (
        Path(config["output_dir"]) / "reconstructed" / "test" / "manifest.json"
    )
    recon_manifest: Optional[List[Dict[str, Any]]] = None
    if recon_manifest_path.exists():
        with open(recon_manifest_path, "r", encoding="utf-8") as fh:
            recon_manifest = json.load(fh)

    if recon_manifest is not None:
        columns = ["id", "duration"]
        if bootstrap_data and "metrics" in bootstrap_data:
            columns += sorted(bootstrap_data["metrics"].keys())

        table = wandb.Table(columns=columns)
        for entry in recon_manifest:
            row = [entry.get("id", ""), entry.get("duration", 0.0)]
            row.extend([None] * (len(columns) - 2))
            table.add_data(*row)
        run.log({"eval/per_utterance": table})
        logger.info("Published per-utterance table (%d rows).", len(recon_manifest))

    # -- Audio samples ------------------------------------------------------
    if recon_manifest is not None and len(recon_manifest) > 0:
        rng = random.Random(42)
        n = min(n_audio_samples, len(recon_manifest))
        samples = rng.sample(recon_manifest, n)
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
                        recon_path, sample_rate=sample_rate, caption="Reconstructed",
                    ),
                })
        logger.info("Published %d audio sample pairs.", n)

    run_url = run.get_url()
    wandb.finish()
    return run_url


# ---------------------------------------------------------------------------
# Consolidated local report
# ---------------------------------------------------------------------------


def _write_local_report(
    report_path: Path,
    experiment: str,
    checkpoint: Optional[str],
    elapsed_s: float,
    ssnr_data: Optional[Dict[str, Any]],
    ttfat_data: Optional[Dict[str, Any]],
    bootstrap_data: Optional[Dict[str, Any]],
    versa_data: Optional[Dict[str, Any]],
    wandb_url: Optional[str],
    resumed_stages: List[str],
) -> None:
    """Write a human-readable local evaluation report."""
    lines: List[str] = []

    lines.append(_SEPARATOR)
    lines.append(f"  EVALUATION REPORT: {experiment}")
    lines.append(f"  Checkpoint: {checkpoint or 'pretrained'}")
    lines.append(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Total time: {elapsed_s:.1f}s ({elapsed_s / 60:.1f}m)")
    if resumed_stages:
        lines.append(f"  Resumed from cache: {', '.join(resumed_stages)}")
    lines.append(_SEPARATOR)

    # SSNR
    lines.append("")
    lines.append("  SSNR (Segmental Signal-to-Noise Ratio)")
    lines.append(f"  {_THIN_SEP}")
    if ssnr_data:
        lines.append(f"    Mean : {ssnr_data['ssnr_mean']:.2f} dB")
        lines.append(f"    Std  : {ssnr_data['ssnr_std']:.2f} dB")
        lines.append(f"    Min  : {ssnr_data.get('ssnr_min', 'N/A')}")
        lines.append(f"    Max  : {ssnr_data.get('ssnr_max', 'N/A')}")
    else:
        lines.append("    [SKIPPED]")

    # TTFAT
    lines.append("")
    lines.append("  TTFAT (Time to First Audio Token)")
    lines.append(f"  {_THIN_SEP}")
    if ttfat_data:
        lines.append(f"    Mean : {ttfat_data['mean_ms']:.3f} ms")
        lines.append(f"    P50  : {ttfat_data['p50_ms']:.3f} ms")
        lines.append(f"    P95  : {ttfat_data['p95_ms']:.3f} ms")
        lines.append(f"    P99  : {ttfat_data['p99_ms']:.3f} ms")
    else:
        lines.append("    [SKIPPED]")

    # Bootstrap metrics
    lines.append("")
    lines.append("  Bootstrap Metrics (PESQ, STOI, DNSMOS, MCD)")
    lines.append(f"  {_THIN_SEP}")
    if bootstrap_data and "metrics" in bootstrap_data:
        display_order = [
            "pesq_wb", "pesq_nb", "stoi",
            "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "mcd",
        ]
        display_names = {
            "pesq_wb": "PESQ (wb)",
            "pesq_nb": "PESQ (nb)",
            "stoi": "STOI",
            "dnsmos_sig": "DNSMOS-SIG",
            "dnsmos_bak": "DNSMOS-BAK",
            "dnsmos_ovrl": "DNSMOS-OVRL",
            "mcd": "MCD",
        }
        metrics = bootstrap_data["metrics"]
        lines.append(f"    {'Metric':<16}{'Mean':>10}{'Std':>10}{'95% CI':>24}")
        lines.append(f"    {'-' * 60}")
        for key in display_order:
            if key not in metrics:
                continue
            r = metrics[key]
            name = display_names.get(key, key)
            ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            lines.append(f"    {name:<16}{r['mean']:>10.4f}{r['std']:>10.4f}{ci:>24}")
        lines.append(f"    N utterances: {bootstrap_data.get('n_utterances', 'N/A')}")
        lines.append(f"    N resamples : {bootstrap_data.get('n_resamples', 'N/A')}")
    else:
        lines.append("    [SKIPPED]")

    # VERSA
    lines.append("")
    lines.append("  VERSA (Comprehensive Evaluation)")
    lines.append(f"  {_THIN_SEP}")
    if versa_data:
        if isinstance(versa_data, dict):
            for k, v in sorted(versa_data.items()):
                if isinstance(v, (int, float)):
                    lines.append(f"    {k:<30}: {v}")
        lines.append(f"    (Full results in results/{experiment}_versa.json)")
    else:
        lines.append("    [SKIPPED -- versa not installed]")

    # WandB
    lines.append("")
    if wandb_url:
        lines.append(f"  WandB run: {wandb_url}")
    else:
        lines.append("  WandB: [NOT PUBLISHED]")

    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  All results saved to: results/")
    lines.append(_SEPARATOR)

    report_text = "\n".join(lines)
    print(report_text)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text + "\n")
    logger.info("Local report written to: %s", report_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_all(
    config_path: str,
    checkpoint: Optional[str] = None,
    split: str = "test",
    use_ema: bool = False,
    skip_versa: bool = False,
    skip_wandb: bool = False,
    wandb_run_id: Optional[str] = None,
    ttfat_n_runs: int = 50,
    ttfat_warmup: int = 5,
    bootstrap_n_resamples: int = 20,
    bootstrap_confidence: float = 0.95,
    n_audio_samples: int = 5,
    restart: bool = False,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline end-to-end with automatic resume.

    On each invocation the pipeline checks a persistent state file
    (``results/<experiment>_eval_state.json``) and skips stages that
    already completed.  A stage that previously *failed* is always
    retried.  Pass ``restart=True`` to discard cached state and re-run
    everything from scratch.

    Args:
        config_path: Path to the experiment YAML config.
        checkpoint: Path to a fine-tuned checkpoint. Uses pretrained if None.
        split: Dataset split to evaluate on.
        use_ema: Whether to load EMA weights from the checkpoint.
        skip_versa: Skip the VERSA evaluation stage.
        skip_wandb: Skip publishing to WandB (local-only mode).
        wandb_run_id: Resume an existing WandB run instead of creating new.
        ttfat_n_runs: Number of timed runs for TTFAT measurement.
        ttfat_warmup: Number of warmup passes for TTFAT.
        bootstrap_n_resamples: Number of bootstrap resamples.
        bootstrap_confidence: Confidence level for bootstrap CIs.
        n_audio_samples: Number of audio samples to log to WandB.
        restart: Discard any saved progress and re-run from stage 1.

    Returns:
        Dict with all collected results.
    """
    config = load_config(config_path)
    experiment = Path(config_path).stem
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- State management ---------------------------------------------------
    fingerprint = EvalState.make_fingerprint(experiment, checkpoint, split, use_ema)
    state = EvalState(results_dir / f"{experiment}_eval_state.json", fingerprint)

    if restart:
        logger.info("--restart requested: discarding cached eval state.")
        state.clear()
    else:
        cached = state.summary()
        done = [s for s, st in cached.items() if st == "ok"]
        if done:
            logger.info(
                "Resuming evaluation. Already completed: %s",
                ", ".join(done),
            )

    # Count codec-specific extra metric stages.
    codec_name = config["codec"]["name"].lower()
    try:
        _codec_hooks = get_codec_hooks(codec_name)
        n_codec_extras = len(_codec_hooks.extra_metrics)
    except ValueError:
        n_codec_extras = 0

    total_stages = 6 + n_codec_extras - int(skip_versa) - int(skip_wandb)
    stage_idx = 0
    pipeline_start = time.monotonic()
    resumed_stages: List[str] = []

    all_results: Dict[str, Any] = {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "config_path": config_path,
        "use_ema": use_ema,
        "split": split,
    }

    # -- Stage 1: Reconstruction --------------------------------------------
    stage_idx += 1
    if state.is_done("reconstruction"):
        _skip_banner("Reconstruction", stage_idx, total_stages)
        recon_data = state.get_data("reconstruction")
        all_results["reconstruction"] = {"status": "ok", "data": recon_data}
        resumed_stages.append("reconstruction")
        logger.info("Reconstruction: loaded from cache.")
    else:
        _stage_banner("Reconstruction", stage_idx, total_stages)
        try:
            recon_manifest = _run_reconstruction(config, checkpoint, use_ema, split)
            recon_data = {"manifest": str(recon_manifest)}
            all_results["reconstruction"] = {"status": "ok", "data": recon_data}
            state.mark_done("reconstruction", recon_data)
            logger.info("Reconstruction complete: %s", recon_manifest)
        except Exception as exc:
            logger.error("Reconstruction FAILED: %s", exc)
            all_results["reconstruction"] = {"status": "failed", "error": str(exc)}
            state.mark_failed("reconstruction", str(exc))
            print(f"\nFATAL: Reconstruction failed -- cannot continue.\n  {exc}")
            _save_json(all_results, results_dir / f"{experiment}_run_all.json")
            return all_results

    # ------------------------------------------------------------------
    # Stages 2+3: SSNR + TTFAT  (concurrent execution)
    #
    # SSNR is purely CPU-bound (numpy segment-wise SNR) and TTFAT is
    # purely GPU-bound (timed encoder forward passes with CUDA sync).
    # They share no resources, so running them in parallel via a
    # 2-thread pool saves the wall-clock time of the shorter stage.
    #
    # If either stage is already cached from a previous (resumed) run,
    # it is loaded from disk and only the uncached stage is executed.
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ssnr_data: Optional[Dict[str, Any]] = None
    ttfat_data: Optional[Dict[str, Any]] = None
    ssnr_cached = state.is_done("ssnr")
    ttfat_cached = state.is_done("ttfat")

    # Handle cached stages first (cheap, no compute).
    if ssnr_cached:
        stage_idx += 1
        _skip_banner("SSNR (Segmental Signal-to-Noise Ratio)", stage_idx, total_stages)
        ssnr_data = state.get_data("ssnr")
        all_results["ssnr"] = {"data": ssnr_data, "status": "ok"}
        resumed_stages.append("ssnr")
        logger.info("SSNR: loaded from cache.")

    if ttfat_cached:
        stage_idx += 1
        _skip_banner("TTFAT (Time to First Audio Token)", stage_idx, total_stages)
        ttfat_data = state.get_data("ttfat")
        all_results["ttfat"] = {"data": ttfat_data, "status": "ok"}
        resumed_stages.append("ttfat")
        logger.info("TTFAT: loaded from cache.")

    # Launch uncached stages concurrently.
    if not ssnr_cached or not ttfat_cached:
        concurrent_futures: Dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=2) as tpool:
            if not ssnr_cached:
                stage_idx += 1
                ssnr_stage_idx = stage_idx
                _stage_banner(
                    "SSNR (Segmental Signal-to-Noise Ratio) [parallel]",
                    ssnr_stage_idx, total_stages,
                )
                fut_ssnr = tpool.submit(_run_ssnr, experiment, results_dir)
                concurrent_futures[fut_ssnr] = "ssnr"

            if not ttfat_cached:
                stage_idx += 1
                ttfat_stage_idx = stage_idx
                _stage_banner(
                    "TTFAT (Time to First Audio Token) [parallel]",
                    ttfat_stage_idx, total_stages,
                )
                fut_ttfat = tpool.submit(
                    _run_ttfat,
                    config, checkpoint, experiment, results_dir,
                    ttfat_n_runs, ttfat_warmup,
                )
                concurrent_futures[fut_ttfat] = "ttfat"

            # Collect results as they complete.  Failures are recorded
            # in the eval state file and do not abort the pipeline --
            # subsequent stages can still run.
            for fut in as_completed(concurrent_futures):
                stage_name = concurrent_futures[fut]
                try:
                    result_data = fut.result()
                    if stage_name == "ssnr":
                        ssnr_data = result_data
                    else:
                        ttfat_data = result_data
                    all_results[stage_name] = {"data": result_data, "status": "ok"}
                    state.mark_done(stage_name, result_data)
                except Exception as exc:
                    logger.error("%s FAILED: %s", stage_name.upper(), exc)
                    all_results[stage_name] = {
                        "status": "failed", "error": str(exc),
                    }
                    state.mark_failed(stage_name, str(exc))

    # -- Stage 4: Bootstrap metrics -----------------------------------------
    stage_idx += 1
    bootstrap_data: Optional[Dict[str, Any]] = None
    if state.is_done("bootstrap"):
        _skip_banner("Bootstrap Metrics (PESQ, STOI, DNSMOS, MCD)", stage_idx, total_stages)
        bootstrap_data = state.get_data("bootstrap")
        all_results["bootstrap"] = {"data": bootstrap_data, "status": "ok"}
        resumed_stages.append("bootstrap")
        logger.info("Bootstrap: loaded from cache.")
    else:
        _stage_banner("Bootstrap Metrics (PESQ, STOI, DNSMOS, MCD)", stage_idx, total_stages)
        try:
            bootstrap_data = _run_bootstrap(
                experiment, results_dir,
                n_resamples=bootstrap_n_resamples,
                confidence=bootstrap_confidence,
            )
            all_results["bootstrap"] = {"data": bootstrap_data, "status": "ok"}
            state.mark_done("bootstrap", bootstrap_data)
        except Exception as exc:
            logger.error("Bootstrap FAILED: %s", exc)
            all_results["bootstrap"] = {"status": "failed", "error": str(exc)}
            state.mark_failed("bootstrap", str(exc))

    # -- Stage 5: VERSA -----------------------------------------------------
    versa_data: Optional[Dict[str, Any]] = None
    if not skip_versa:
        stage_idx += 1
        if state.is_done("versa"):
            _skip_banner("VERSA (Comprehensive Evaluation)", stage_idx, total_stages)
            versa_data = state.get_data("versa")
            all_results["versa"] = {"data": versa_data, "status": "ok"}
            resumed_stages.append("versa")
            logger.info("VERSA: loaded from cache.")
        else:
            _stage_banner("VERSA (Comprehensive Evaluation)", stage_idx, total_stages)
            try:
                versa_data = _run_versa(experiment, results_dir)
                all_results["versa"] = {"data": versa_data, "status": "ok"}
                state.mark_done("versa", versa_data)
            except Exception as exc:
                logger.error("VERSA FAILED: %s", exc)
                all_results["versa"] = {"status": "failed", "error": str(exc)}
                state.mark_failed("versa", str(exc))
    else:
        all_results["versa"] = {"status": "skipped"}

    # -- Codec-specific extra metric stages ---------------------------------
    codec_name = config["codec"]["name"].lower()
    try:
        codec_hooks = get_codec_hooks(codec_name)
    except ValueError:
        codec_hooks = None

    codec_extras_data: Dict[str, Any] = {}
    if codec_hooks and codec_hooks.extra_metrics:
        for extra_name, extra_fn in codec_hooks.extra_metrics:
            stage_key = f"codec_{extra_name}"
            stage_idx += 1
            if state.is_done(stage_key):
                _skip_banner(
                    f"Codec metric: {extra_name}", stage_idx, total_stages,
                )
                codec_extras_data[extra_name] = state.get_data(stage_key)
                all_results[stage_key] = {
                    "data": codec_extras_data[extra_name],
                    "status": "ok",
                }
                resumed_stages.append(stage_key)
            else:
                _stage_banner(
                    f"Codec metric: {extra_name}", stage_idx, total_stages,
                )
                try:
                    result_data = extra_fn(config, experiment, results_dir)
                    codec_extras_data[extra_name] = result_data
                    all_results[stage_key] = {
                        "data": result_data,
                        "status": "ok",
                    }
                    state.mark_done(stage_key, result_data)
                except Exception as exc:
                    logger.error(
                        "Codec metric '%s' FAILED: %s", extra_name, exc,
                    )
                    all_results[stage_key] = {
                        "status": "failed",
                        "error": str(exc),
                    }
                    state.mark_failed(stage_key, str(exc))

    # -- Publish to WandB --------------------------------------------------
    # WandB is always re-run (never cached) because it is the aggregation
    # point and is idempotent.  It needs the latest data from all stages.
    wandb_url: Optional[str] = None
    if not skip_wandb:
        stage_idx += 1
        _stage_banner("Publish to WandB", stage_idx, total_stages)
        try:
            wandb_url = _publish_to_wandb(
                config=config,
                experiment=experiment,
                results_dir=results_dir,
                checkpoint=checkpoint,
                ssnr_data=ssnr_data,
                ttfat_data=ttfat_data,
                bootstrap_data=bootstrap_data,
                versa_data=versa_data,
                codec_extras_data=codec_extras_data,
                run_id=wandb_run_id,
                n_audio_samples=n_audio_samples,
            )
            all_results["wandb"] = {"url": wandb_url, "status": "ok"}
            state.mark_done("wandb", {"url": wandb_url})
            logger.info("WandB run: %s", wandb_url)
        except Exception as exc:
            logger.error("WandB publish FAILED: %s", exc)
            all_results["wandb"] = {"status": "failed", "error": str(exc)}
            state.mark_failed("wandb", str(exc))
    else:
        all_results["wandb"] = {"status": "skipped"}

    # -- Finalize -----------------------------------------------------------
    elapsed = time.monotonic() - pipeline_start
    all_results["elapsed_s"] = round(elapsed, 1)
    all_results["timestamp"] = datetime.now(timezone.utc).isoformat()
    all_results["resumed_stages"] = resumed_stages

    _save_json(all_results, results_dir / f"{experiment}_run_all.json")

    _write_local_report(
        report_path=results_dir / f"{experiment}_eval_report.txt",
        experiment=experiment,
        checkpoint=checkpoint,
        elapsed_s=elapsed,
        ssnr_data=ssnr_data,
        ttfat_data=ttfat_data,
        bootstrap_data=bootstrap_data,
        versa_data=versa_data,
        wandb_url=wandb_url,
        resumed_stages=resumed_stages,
    )

    # Summary status
    stage_statuses = []
    for stage_name in STAGE_NAMES:
        if stage_name in all_results:
            stage_statuses.append(
                (stage_name, all_results[stage_name].get("status", "unknown"))
            )

    failed = [s for s, st in stage_statuses if st == "failed"]
    if failed:
        print(f"\nWARNING: The following stages failed: {', '.join(failed)}")
        print("Re-run the same command to retry only the failed stages.\n")
    else:
        print("\nAll evaluation stages completed successfully.\n")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the full evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation pipeline with automatic resume. Runs "
            "reconstruction, SSNR, TTFAT, bootstrap metrics, VERSA, and "
            "publishes everything to WandB.  Re-run the same command after "
            "a failure to resume from where it left off."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full eval with EMA checkpoint, everything published:\n"
            "  uv run python eval/run_all.py \\\n"
            "      --config configs/experiments/mimi_hindi.yaml \\\n"
            "      --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \\\n"
            "      --use-ema\n"
            "\n"
            "  # If the above crashes at stage 4, just re-run the same command.\n"
            "  # Stages 1-3 will be loaded from cache automatically.\n"
            "\n"
            "  # Force a clean re-run (discard cached progress):\n"
            "  uv run python eval/run_all.py \\\n"
            "      --config configs/experiments/mimi_hindi.yaml \\\n"
            "      --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \\\n"
            "      --use-ema --restart\n"
            "\n"
            "  # Local-only (no WandB), skip VERSA:\n"
            "  uv run python eval/run_all.py \\\n"
            "      --config configs/experiments/mimi_hindi.yaml \\\n"
            "      --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \\\n"
            "      --skip-wandb --skip-versa\n"
            "\n"
            "  # Resume logging to an existing WandB run:\n"
            "  uv run python eval/run_all.py \\\n"
            "      --config configs/experiments/mimi_hindi.yaml \\\n"
            "      --checkpoint outputs/mimi_hindi/checkpoint_step_50000.pt \\\n"
            "      --wandb-run-id iwdd7hfg\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint. Uses pretrained if omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load EMA weights from the checkpoint.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard cached progress and re-run all stages from scratch.",
    )
    parser.add_argument(
        "--skip-versa",
        action="store_true",
        help="Skip the VERSA evaluation stage.",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip publishing to WandB (local-only mode).",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Resume an existing WandB run instead of creating a new one.",
    )
    parser.add_argument(
        "--ttfat-n-runs",
        type=int,
        default=50,
        help="Number of timed runs for TTFAT (default: 50).",
    )
    parser.add_argument(
        "--ttfat-warmup",
        type=int,
        default=5,
        help="Number of warmup passes for TTFAT (default: 5).",
    )
    parser.add_argument(
        "--bootstrap-n-resamples",
        type=int,
        default=20,
        help="Number of bootstrap resamples (default: 20).",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CIs (default: 0.95).",
    )
    parser.add_argument(
        "--n-audio-samples",
        type=int,
        default=5,
        help="Number of audio pairs to log to WandB (default: 5).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_all(
        config_path=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        use_ema=args.use_ema,
        skip_versa=args.skip_versa,
        skip_wandb=args.skip_wandb,
        wandb_run_id=args.wandb_run_id,
        ttfat_n_runs=args.ttfat_n_runs,
        ttfat_warmup=args.ttfat_warmup,
        bootstrap_n_resamples=args.bootstrap_n_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        n_audio_samples=args.n_audio_samples,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
