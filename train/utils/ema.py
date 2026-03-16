"""Exponential Moving Average (EMA) model wrapper.

Maintains a shadow copy of model parameters and updates them as an
exponential moving average of the training weights.  During evaluation
the EMA weights can be temporarily swapped in for improved generation
quality, then restored after.

Usage::

    from train.utils.ema import EMAModel

    ema = EMAModel(model, decay=0.999, start_step=1000)
    for step, batch in enumerate(loader):
        loss = train_step(model, batch)
        ema.update(model, step)

    # Evaluate with EMA weights.
    ema.apply_to(model)
    evaluate(model)
    ema.restore(model)

Dependencies:
    torch

License: MIT
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn


class EMAModel:
    """Shadow-parameter EMA tracker.

    Args:
        model: The model whose parameters will be tracked.
        decay: EMA decay factor.  Higher values produce smoother averages.
        start_step: Training step at which EMA updates begin.  Before this
            step, shadow parameters are **not** updated (the shadow stays
            equal to the initial weights).

    Example::

        >>> model = nn.Linear(32, 16)
        >>> ema = EMAModel(model, decay=0.999, start_step=500)
        >>> ema.update(model, step=1000)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        start_step: int = 1000,
    ) -> None:
        self.decay = decay
        self.start_step = start_step

        # Deep-copy the initial parameters as the EMA shadow.
        self._shadow: Dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        # Backup slot: populated by ``apply_to`` and consumed by ``restore``.
        self._backup: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Update EMA shadow parameters from *model*.

        If *step* is below ``start_step`` the call is a no-op, allowing
        the model to stabilise before averaging begins.

        Args:
            model: The training model whose current parameters are used
                for the EMA update.
            step: The current global training step.
        """
        if step < self.start_step:
            return

        d = self.decay
        for name, param in model.named_parameters():
            # EMA formula: shadow = decay * shadow + (1 - decay) * param
            self._shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the EMA shadow parameters as a serialisable dict.

        Returns:
            A dictionary mapping parameter names to their EMA values.
        """
        return {k: v.clone() for k, v in self._shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA shadow parameters from a checkpoint.

        Args:
            state_dict: A dictionary previously returned by
                :meth:`state_dict`.

        Raises:
            KeyError: If *state_dict* is missing expected parameter names.
        """
        for name in self._shadow:
            if name not in state_dict:
                raise KeyError(f"Missing EMA parameter: {name!r}")
            self._shadow[name].copy_(state_dict[name])

    # ------------------------------------------------------------------
    # Swap helpers (for evaluation)
    # ------------------------------------------------------------------

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into *model* for evaluation.

        The current (training) weights are stored in an internal backup so
        they can be restored later with :meth:`restore`.

        Args:
            model: The model to receive EMA weights.
        """
        self._backup = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        for name, param in model.named_parameters():
            param.data.copy_(self._shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original training weights after an EMA evaluation pass.

        Args:
            model: The model previously modified by :meth:`apply_to`.

        Raises:
            RuntimeError: If called without a prior :meth:`apply_to`.
        """
        if not self._backup:
            raise RuntimeError("restore() called without a prior apply_to()")
        for name, param in model.named_parameters():
            param.data.copy_(self._backup[name])
        self._backup = {}
