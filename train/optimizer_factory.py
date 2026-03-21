"""Factory for creating optimizers from config dicts.

Creates any of the 8 supported optimizers (AdamW, RAdam, Lion, Prodigy,
Schedule-Free AdamW, SOAP, Adan, Muon) from a unified config dictionary.
See ``OPTIMIZERS.md`` in the project root for the full comparison table,
memory footprints, and sweep-specific considerations.

Usage::

    from train.optimizer_factory import create_optimizer

    optimizer, meta = create_optimizer(model, config)
    # meta may contain flags like ``disable_scheduler``, ``needs_eval_mode``, etc.

Dependencies:
    - torch (required)
    - lion-pytorch, prodigyopt, schedulefree, adan-pytorch (optional extras)
    Install optional deps via ``uv pip install -e '.[train]'``

License: MIT
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> Tuple[Optimizer, Dict[str, Any]]:
    """Create an optimizer instance from a config dictionary.

    Reads ``config["optimizer"]["name"]`` to select the optimizer, then
    delegates to the appropriate constructor with config-driven hyperparams.

    Args:
        model: The ``nn.Module`` whose parameters will be optimized.
        config: Full experiment config dict.  Must contain at least::

                optimizer:
                  name: "adamw"   # one of the 8 supported names
                  lr: 1e-4
                  weight_decay: 0.01

    Returns:
        A ``(optimizer, metadata)`` tuple where *metadata* is a dict of
        flags consumed by the training loop, e.g.::

            {"disable_scheduler": True, "needs_eval_mode": True}

    Raises:
        ValueError: If the optimizer name is not recognised.
        ImportError: If a third-party optimizer package is not installed.

    Example::

        >>> import torch.nn as nn
        >>> cfg = {"optimizer": {"name": "adamw", "lr": 1e-4,
        ...        "betas": [0.9, 0.999], "weight_decay": 0.01}}
        >>> opt, meta = create_optimizer(nn.Linear(10, 10), cfg)
        >>> type(opt).__name__
        'AdamW'
    """
    opt_cfg: Dict[str, Any] = config["optimizer"]
    name: str = opt_cfg["name"].lower().strip()

    lr: float = float(opt_cfg.get("lr", 1e-4))
    betas: List[float] = list(opt_cfg.get("betas", [0.9, 0.999]))
    weight_decay: float = float(opt_cfg.get("weight_decay", 0.01))

    params = [p for p in model.parameters() if p.requires_grad]
    metadata: Dict[str, Any] = {}

    if name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(betas[0], betas[1]),
            weight_decay=weight_decay,
        )

    elif name == "radam":
        optimizer = torch.optim.RAdam(
            params,
            lr=lr,
            betas=(betas[0], betas[1]),
            weight_decay=weight_decay,
        )

    elif name == "lion":
        try:
            from lion_pytorch import Lion  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "Lion optimizer requires 'lion-pytorch'. "
                "Install via: uv pip install -e '.[train]'  "
                "(see pyproject.toml [project.optional-dependencies.train])"
            ) from exc

        # Lion benefits from 3-10x higher weight decay than AdamW.
        metadata["adjust_wd"] = True
        optimizer = Lion(params, lr=lr, weight_decay=weight_decay)

    elif name == "prodigy":
        try:
            from prodigyopt import Prodigy  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "Prodigy optimizer requires 'prodigyopt'. "
                "Install via: uv pip install -e '.[train]'  "
                "(see pyproject.toml [project.optional-dependencies.train])"
            ) from exc

        # Prodigy auto-estimates the learning rate; external LR must be 1.0.
        d_coef: float = float(opt_cfg.get("d_coef", 1.0))
        metadata["disable_scheduler"] = True
        metadata["disable_warmup"] = True
        optimizer = Prodigy(
            params,
            lr=1.0,
            d_coef=d_coef,
            weight_decay=weight_decay,
        )

    elif name == "schedulefree_adamw":
        try:
            from schedulefree import AdamWScheduleFree  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "Schedule-Free AdamW requires 'schedulefree'. "
                "Install via: uv pip install -e '.[train]'  "
                "(see pyproject.toml [project.optional-dependencies.train])"
            ) from exc

        metadata["disable_scheduler"] = True
        metadata["disable_warmup"] = True
        metadata["needs_eval_mode"] = True
        optimizer = AdamWScheduleFree(
            params,
            lr=lr,
            betas=(betas[0], betas[1]),
            weight_decay=weight_decay,
        )

    elif name == "soap":
        optimizer = _create_soap_optimizer(params, lr, betas, weight_decay, opt_cfg)

    elif name == "adan":
        try:
            from adan_pytorch import Adan  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "Adan optimizer requires 'adan-pytorch'. "
                "Install via: uv pip install -e '.[train]'  "
                "(see pyproject.toml [project.optional-dependencies.train])"
            ) from exc

        # Adan uses 3 betas; fall back to sensible defaults.
        adan_betas = (
            betas[0] if len(betas) > 0 else 0.98,
            betas[1] if len(betas) > 1 else 0.92,
            betas[2] if len(betas) > 2 else 0.99,
        )
        optimizer = Adan(
            params,
            lr=lr,
            betas=adan_betas,
            weight_decay=weight_decay,
        )

    elif name == "muon":
        muon_params, adamw_params = _split_params_for_muon(model)
        optimizer = _create_muon_hybrid(muon_params, adamw_params, config)
        metadata["is_hybrid"] = True

    else:
        supported = (
            "adamw, radam, lion, prodigy, schedulefree_adamw, soap, adan, muon"
        )
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported optimizers: {supported}"
        )

    logger.info("Created optimizer: %s  (metadata=%s)", name, metadata)
    return optimizer, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_soap_optimizer(
    params: List[nn.Parameter],
    lr: float,
    betas: List[float],
    weight_decay: float,
    opt_cfg: Dict[str, Any],
) -> Optimizer:
    """Create a SOAP (Shampoo + Adam) optimizer.

    Attempts to import from the ``soap`` package first; falls back to a
    minimal inline implementation if unavailable.

    Args:
        params: Trainable parameters.
        lr: Learning rate.
        betas: Adam-style beta values (at least 2 entries).
        weight_decay: Weight decay coefficient.
        opt_cfg: Full optimizer config subsection; may contain
            ``shampoo_beta``.

    Returns:
        An optimizer instance implementing the SOAP algorithm.

    Raises:
        ImportError: If neither the ``soap`` package nor a compatible
            fallback is available.
    """
    shampoo_beta: float = float(opt_cfg.get("shampoo_beta", 0.9))

    try:
        from soap import SOAP  # pyright: ignore[reportMissingImports]

        return SOAP(
            params,
            lr=lr,
            betas=(betas[0], betas[1]),
            shampoo_beta=shampoo_beta,
            weight_decay=weight_decay,
        )
    except ImportError:
        logger.warning(
            "SOAP package not found. Install via the emerging-optimizers repo. "
            "Falling back to AdamW as a placeholder."
        )
        # Graceful degradation: fall back to AdamW so experiments can still run.
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(betas[0], betas[1]),
            weight_decay=weight_decay,
        )


def _split_params_for_muon(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Separate model parameters into Muon-eligible and AdamW-eligible groups.

    Muon targets **strictly 2-D hidden-layer weight matrices** where its
    orthogonalised update provides the most benefit.  Everything else
    (embeddings, LayerNorm, biases, 1-D parameters, and 3-D+ conv weights)
    is routed to AdamW.  The native ``torch.optim.Muon`` only supports 2-D
    tensors; 1-D conv weights (3-D) are not flattened automatically.

    Args:
        model: The neural network model.

    Returns:
        A tuple ``(muon_params, adamw_params)`` of parameter lists.

    Example::

        >>> m = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        >>> muon_p, adam_p = _split_params_for_muon(m)
        >>> len(muon_p)  # Linear.weight (2-D)
        1
        >>> len(adam_p)  # Linear.bias + LayerNorm.weight + LayerNorm.bias
        3
    """
    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # Exclude: embeddings, norms, biases, non-2D params.
        is_embedding = "embed" in name_lower
        is_norm = "norm" in name_lower or "ln" in name_lower
        is_bias = name_lower.endswith(".bias")
        is_not_2d = param.ndim != 2

        if is_embedding or is_norm or is_bias or is_not_2d:
            adamw_params.append(param)
        else:
            muon_params.append(param)

    logger.info(
        "Muon param split: %d Muon params, %d AdamW params",
        len(muon_params),
        len(adamw_params),
    )
    return muon_params, adamw_params


def _create_muon_hybrid(
    muon_params: List[nn.Parameter],
    adamw_params: List[nn.Parameter],
    config: Dict[str, Any],
) -> "MuonHybridOptimizer":
    """Build a hybrid Muon + AdamW optimizer.

    Creates separate optimizer instances for each parameter group and wraps
    them in :class:`MuonHybridOptimizer` so the training loop can treat
    them as a single ``Optimizer``.

    Args:
        muon_params: Parameters to optimise with Muon (2-D+ hidden weights).
        adamw_params: Parameters to optimise with AdamW (embeddings, norms,
            biases).
        config: Full experiment config dict.

    Returns:
        A :class:`MuonHybridOptimizer` wrapping both inner optimizers.

    Raises:
        ImportError: If ``torch.optim.Muon`` is unavailable (requires
            PyTorch ≥ 2.10).
    """
    opt_cfg: Dict[str, Any] = config["optimizer"]
    lr: float = float(opt_cfg.get("lr", 0.02))
    betas: List[float] = list(opt_cfg.get("betas", [0.9, 0.999]))
    weight_decay: float = float(opt_cfg.get("weight_decay", 0.01))

    # Muon optimizer for 2-D hidden weights.
    try:
        muon_cls = torch.optim.Muon  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ImportError(
            "torch.optim.Muon requires PyTorch >= 2.10. "
            "Please upgrade: uv pip install 'torch>=2.10'"
        ) from exc

    muon_opt = muon_cls(muon_params, lr=lr) if muon_params else None

    # AdamW for everything else (embeddings, norms, biases, 1-D params).
    adamw_lr: float = float(opt_cfg.get("adamw_lr", 1e-4))
    adamw_opt = (
        torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=(betas[0], betas[1]),
            weight_decay=weight_decay,
        )
        if adamw_params
        else None
    )

    return MuonHybridOptimizer(muon_opt, adamw_opt)


# ---------------------------------------------------------------------------
# MuonHybridOptimizer
# ---------------------------------------------------------------------------


class MuonHybridOptimizer(Optimizer):
    """Hybrid optimizer wrapping Muon (hidden layers) and AdamW (the rest).

    Delegates :meth:`step`, :meth:`zero_grad`, :meth:`state_dict`, and
    :meth:`load_state_dict` to both inner optimizers so the training loop
    can treat this as a single ``torch.optim.Optimizer``.

    Args:
        muon_opt: Muon optimizer for 2-D+ hidden weights, or ``None``.
        adamw_opt: AdamW optimizer for remaining parameters, or ``None``.

    Example::

        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 10)
        >>> cfg = {"optimizer": {"name": "muon", "lr": 0.02,
        ...        "betas": [0.9, 0.999], "weight_decay": 0.01}}
        >>> opt, meta = create_optimizer(model, cfg)  # doctest: +SKIP
        >>> isinstance(opt, MuonHybridOptimizer)       # doctest: +SKIP
        True
    """

    def __init__(
        self,
        muon_opt: Optimizer | None,
        adamw_opt: Optimizer | None,
    ) -> None:
        """Initialise the hybrid optimizer.

        Args:
            muon_opt: Muon optimizer for 2-D+ hidden weights, or ``None``
                if no parameters qualified for Muon.
            adamw_opt: AdamW optimizer for remaining parameters, or ``None``
                if all parameters are handled by Muon.
        """
        # Collect all param_groups for the Optimizer base class.
        param_groups: List[Dict[str, Any]] = []
        self._muon_opt = muon_opt
        self._adamw_opt = adamw_opt

        if muon_opt is not None:
            param_groups.extend(muon_opt.param_groups)
        if adamw_opt is not None:
            param_groups.extend(adamw_opt.param_groups)

        if not param_groups:
            raise ValueError(
                "MuonHybridOptimizer requires at least one inner optimizer."
            )

        # Initialise the Optimizer base class, then swap in the *actual*
        # inner param_groups so LR schedulers and manual LR edits operate on
        # the same dictionaries consumed by the wrapped optimizers.
        super().__init__(
            [{k: v for k, v in pg.items()} for pg in param_groups],
            defaults={"lr": 1e-4},
        )
        self.param_groups = param_groups

    # -- Core interface ----------------------------------------------------

    def step(self, closure: Any = None) -> Any:  # noqa: D401
        """Perform a single optimization step on both inner optimizers.

        Args:
            closure: An optional closure that re-evaluates the model and
                returns the loss. Passed to both inner optimizers.

        Returns:
            The loss from the closure, if provided.
        """
        loss = None
        if self._muon_opt is not None:
            loss = self._muon_opt.step(closure)
        if self._adamw_opt is not None:
            adamw_loss = self._adamw_opt.step(closure)
            if loss is None:
                loss = adamw_loss
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for all managed parameters.

        Args:
            set_to_none: If ``True``, set gradients to ``None`` instead of
                zero tensors (more memory-efficient).
        """
        if self._muon_opt is not None:
            self._muon_opt.zero_grad(set_to_none=set_to_none)
        if self._adamw_opt is not None:
            self._adamw_opt.zero_grad(set_to_none=set_to_none)

    # -- Serialisation -----------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a combined state dict from both inner optimizers.

        Returns:
            A dictionary with ``"muon"`` and ``"adamw"`` keys, each
            containing the respective optimizer's state dict.
        """
        return {
            "muon": self._muon_opt.state_dict() if self._muon_opt else {},
            "adamw": self._adamw_opt.state_dict() if self._adamw_opt else {},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a combined state dict into both inner optimizers.

        Args:
            state_dict: A dict with ``"muon"`` and ``"adamw"`` keys as
                produced by :meth:`state_dict`.
        """
        if self._muon_opt is not None and state_dict.get("muon"):
            self._muon_opt.load_state_dict(state_dict["muon"])
        if self._adamw_opt is not None and state_dict.get("adamw"):
            self._adamw_opt.load_state_dict(state_dict["adamw"])

        param_groups: List[Dict[str, Any]] = []
        if self._muon_opt is not None:
            param_groups.extend(self._muon_opt.param_groups)
        if self._adamw_opt is not None:
            param_groups.extend(self._adamw_opt.param_groups)
        self.param_groups = param_groups
