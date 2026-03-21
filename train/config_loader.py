"""YAML config loading with hierarchical ``_bases_`` merging and validation.

Loads experiment YAML configs, recursively resolves ``_bases_`` references
(each base can itself reference further bases), deep-merges them in order,
validates required fields, and optionally applies WandB sweep overrides.

Design
------
Experiment configs follow a layered inheritance pattern:

1. **Base configs** (e.g. ``configs/codecs/mimi.yaml``) define
   codec-specific defaults (sample rate, latency, loss weights).
2. **Dataset configs** (e.g. ``configs/datasets/turkish_sample.yaml``)
   define data paths, language, and split information.
3. **Experiment configs** (e.g.
   ``configs/experiments/mimi_turkish_sample.yaml``) list their bases
   via the ``_bases_`` key and add experiment-specific overrides.

Merging is performed recursively and left-to-right: later bases override
earlier ones, and the experiment's own keys override everything.

Sweep integration
-----------------
:func:`apply_sweep_overrides` maps flat WandB sweep parameter names
(e.g. ``learning_rate``, ``beta1``) to their nested config paths
(``optimizer.lr``, ``optimizer.betas[0]``).  It also applies
optimizer-specific adjustments (e.g. forcing ``scheduler="constant"``
for Prodigy, boosting ``weight_decay`` for Lion).

Usage::

    from train.config_loader import load_config, apply_sweep_overrides

    config = load_config("configs/experiments/mimi_turkish_sample.yaml")
    config = apply_sweep_overrides(config, sweep_params)

Dependencies:
    - pyyaml (required, listed in core dependencies)

License: MIT
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required config field paths -- each dot-separated string is a nested key.
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: list[str] = [
    "optimizer.name",
    "training.max_steps",
    "training.seed",
    "codec.name",
    "codec.pretrained",
    "codec.sample_rate",
    "dataset.name",
    "dataset.language",
    "output_dir",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file with recursive ``_bases_`` resolution.

    The merging order is:

    1. Each file listed in ``_bases_`` is loaded (and its own bases resolved
       recursively).
    2. Bases are merged left-to-right — later bases override earlier ones.
    3. The experiment's own keys override everything.

    After merging, the result is validated via :func:`validate_config`.

    Args:
        config_path: Path to the experiment YAML file. May be a string or
            :class:`~pathlib.Path`.

    Returns:
        The fully merged and validated config dictionary.

    Raises:
        FileNotFoundError: If the config file (or any base) does not exist.
        yaml.YAMLError: If any YAML file is malformed.
        ValueError: If required config fields are missing after merging.

    Example::

        >>> config = load_config("configs/experiments/mimi_turkish_sample.yaml")
        >>> config["optimizer"]["name"]
        'adamw'
    """
    config_path = Path(config_path).resolve()
    logger.info("Loading config: %s", config_path)

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    config_dir = config_path.parent
    config = _resolve_bases(raw, config_dir)

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Check that all required fields are present in the config.

    Required fields are defined by the ``_REQUIRED_FIELDS`` module constant
    and cover optimizer, training, codec, dataset, and output settings.

    Args:
        config: The merged config dictionary to validate.

    Raises:
        ValueError: With a descriptive message listing every missing field.

    Example::

        >>> validate_config({"optimizer": {"name": "adamw"}})
        Traceback (most recent call last):
            ...
        ValueError: Config is missing required fields: ...
    """
    missing: list[str] = []
    for field_path in _REQUIRED_FIELDS:
        keys = field_path.split(".")
        node: Any = config
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                missing.append(field_path)
                break
            node = node[key]

    if missing:
        raise ValueError(
            f"Config is missing required fields: {', '.join(missing)}. "
            "Check your experiment YAML and ensure all bases are merged correctly."
        )


def apply_sweep_overrides(
    config: Dict[str, Any],
    sweep_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply WandB sweep parameters onto a base config.

    Handles optimizer-specific logic:

    * **Prodigy / Schedule-Free**: sets scheduler to ``"constant"``
      (effectively disabling external scheduling).
    * **Lion**: if the sweep's ``weight_decay`` is below ``0.03``, it is
      multiplied by 10 to match Lion's recommended 3-10× higher range.

    Flat sweep parameter names are mapped to nested config keys:

    ============== ============================
    Sweep key      Config path
    ============== ============================
    learning_rate  optimizer.lr
    weight_decay   optimizer.weight_decay
    optimizer      optimizer.name
    beta1          optimizer.betas[0]
    beta2          optimizer.betas[1]
    scheduler      scheduler.name
    warmup_steps   scheduler.warmup_steps
    ============== ============================

    Args:
        config: The base config (will be deep-copied, not mutated).
        sweep_params: Flat dictionary of parameter values from a WandB
            sweep run.

    Returns:
        A new config dict with sweep overrides applied.

    Example::

        >>> base = {"optimizer": {"name": "adamw", "lr": 1e-4,
        ...         "weight_decay": 0.01, "betas": [0.9, 0.999]},
        ...         "scheduler": {"name": "cosine", "warmup_steps": 500}}
        >>> swept = apply_sweep_overrides(base, {"learning_rate": 3e-4})
        >>> swept["optimizer"]["lr"]
        0.0003
    """
    cfg = copy.deepcopy(config)

    # ── Map flat sweep keys → nested config locations ──────────────────
    # Optimizer core hyper-parameters.
    if "optimizer" in sweep_params:
        cfg["optimizer"]["name"] = sweep_params["optimizer"]

    if "learning_rate" in sweep_params:
        cfg["optimizer"]["lr"] = float(sweep_params["learning_rate"])

    if "weight_decay" in sweep_params:
        cfg["optimizer"]["weight_decay"] = float(sweep_params["weight_decay"])

    # Beta values are stored as a 2-element list; setdefault ensures the
    # list exists before indexing into it.
    if "beta1" in sweep_params:
        cfg["optimizer"].setdefault("betas", [0.9, 0.999])
        cfg["optimizer"]["betas"][0] = float(sweep_params["beta1"])

    if "beta2" in sweep_params:
        cfg["optimizer"].setdefault("betas", [0.9, 0.999])
        cfg["optimizer"]["betas"][1] = float(sweep_params["beta2"])

    # Scheduler settings.
    if "scheduler" in sweep_params:
        cfg.setdefault("scheduler", {})["name"] = sweep_params["scheduler"]

    if "warmup_steps" in sweep_params:
        cfg.setdefault("scheduler", {})["warmup_steps"] = int(
            sweep_params["warmup_steps"]
        )

    # ── Training-level parameters ────────────────────────────────────────
    if "max_steps" in sweep_params:
        cfg["training"]["max_steps"] = int(sweep_params["max_steps"])

    if "segment_s" in sweep_params:
        cfg["codec"]["training"]["segment_s"] = float(sweep_params["segment_s"])

    if "augmentation_preset" in sweep_params:
        cfg.setdefault("augmentation", {})["preset"] = sweep_params["augmentation_preset"]

    # ── Discriminator parameters ─────────────────────────────────────────
    if "disc_lr_ratio" in sweep_params:
        cfg.setdefault("discriminator", {})["lr_ratio"] = float(sweep_params["disc_lr_ratio"])

    if "r1_penalty" in sweep_params:
        cfg.setdefault("discriminator", {})["r1_penalty"] = float(sweep_params["r1_penalty"])

    if "disc_warmup_steps" in sweep_params:
        cfg.setdefault("discriminator", {})["warmup_steps"] = int(sweep_params["disc_warmup_steps"])

    # ── EMA ──────────────────────────────────────────────────────────────
    if "ema_decay" in sweep_params:
        cfg.setdefault("ema", {})["decay"] = float(sweep_params["ema_decay"])

    # ── Encoder freezing ─────────────────────────────────────────────────
    if "freeze_encoder_steps" in sweep_params:
        cfg["codec"]["training"]["freeze_encoder_steps"] = int(sweep_params["freeze_encoder_steps"])

    # --- Optimizer-specific adjustments ---

    opt_name = cfg["optimizer"]["name"].lower().strip()

    if opt_name in ("prodigy", "schedulefree_adamw"):
        # These optimizers handle scheduling internally.
        cfg.setdefault("scheduler", {})["name"] = "constant"
        logger.info(
            "Optimizer '%s' detected — forcing scheduler to 'constant'.",
            opt_name,
        )

    if opt_name == "lion":
        wd = cfg["optimizer"].get("weight_decay", 0.01)
        if wd < 0.03:
            cfg["optimizer"]["weight_decay"] = wd * 10
            logger.info(
                "Lion optimizer: weight_decay %.4f < 0.03 — "
                "multiplied by 10 → %.4f (Lion needs 3-10x higher wd).",
                wd,
                cfg["optimizer"]["weight_decay"],
            )

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into *base*, returning a new dict.

    * Nested dicts are merged recursively.
    * Lists are **replaced** (not appended).
    * Scalar values in *override* take precedence.

    Args:
        base: The base config dictionary.
        override: The overriding config dictionary.

    Returns:
        A new dictionary containing the merged result.

    Example::

        >>> _merge_configs(
        ...     {"a": 1, "b": {"c": 2, "d": 3}},
        ...     {"b": {"c": 99}},
        ... )
        {'a': 1, 'b': {'c': 99, 'd': 3}}
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_bases(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """Recursively resolve ``_bases_`` references and merge them.

    Each entry in ``_bases_`` is a relative path (resolved against
    *config_dir*).  Bases are loaded and merged left-to-right, and the
    current file's own keys override all bases.

    Args:
        config: Raw config dict (may contain a ``_bases_`` key).
        config_dir: Directory of the config file, used to resolve relative
            paths.

    Returns:
        Fully merged config with the ``_bases_`` key removed.

    Raises:
        FileNotFoundError: If a referenced base file does not exist.
    """
    bases: list[str] = config.pop("_bases_", [])
    if not bases:
        return config

    merged: Dict[str, Any] = {}
    for base_relpath in bases:
        base_path = (config_dir / base_relpath).resolve()
        logger.debug("Resolving base: %s", base_path)

        with open(base_path, "r", encoding="utf-8") as fh:
            base_raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # Recursively resolve the base's own _bases_.
        base_resolved = _resolve_bases(base_raw, base_path.parent)
        merged = _merge_configs(merged, base_resolved)

    # The experiment's own keys override all bases.
    merged = _merge_configs(merged, config)
    return merged
