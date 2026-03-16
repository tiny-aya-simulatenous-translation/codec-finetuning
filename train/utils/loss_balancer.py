"""AudioCraft-style gradient-normalised loss balancer.

When training with multiple loss terms (reconstruction, adversarial,
feature-matching, commitment, …) the loss with the largest gradient
magnitude can dominate the shared encoder/decoder updates, starving
other objectives.  This module rescales each loss so that every term
contributes **equally** in gradient magnitude to the shared output
tensor.

Usage::

    from train.utils.loss_balancer import LossBalancer

    balancer = LossBalancer(weights={"rec": 1.0, "adv": 0.1, "fm": 2.0})
    total = balancer.balance(losses, shared_output=encoder_output)
    total.backward()

Dependencies:
    torch

License: MIT
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


class LossBalancer:
    """Rescale multiple losses to equalise their gradient norms.

    Maintains an exponential moving average (EMA) of each loss's gradient
    norm with respect to a shared output tensor, then divides each loss
    by its EMA norm before weighting and summing.

    Args:
        weights: Mapping from loss name to its desired relative weight.
        ema_decay: Decay factor for the gradient-norm EMA.  Values close
            to 1.0 provide a smoother (slower-adapting) estimate.

    Raises:
        ValueError: If *weights* is empty.

    Example::

        >>> balancer = LossBalancer({"rec": 1.0, "adv": 0.1})
        >>> total = balancer.balance(
        ...     {"rec": rec_loss, "adv": adv_loss},
        ...     shared_output=z,
        ... )
    """

    def __init__(self, weights: Dict[str, float], ema_decay: float = 0.999) -> None:
        if not weights:
            raise ValueError("weights must contain at least one entry")
        self.weights = dict(weights)
        self.ema_decay = ema_decay
        # EMA of per-loss gradient norms.  Initialised lazily on first call.
        self._ema_norms: Dict[str, float] = {}

    def balance(
        self,
        losses: Dict[str, Tensor],
        shared_output: Tensor,
    ) -> Tensor:
        """Compute the gradient-balanced combined loss.

        For each loss term the method:

        1. Computes the gradient of that loss w.r.t. *shared_output*.
        2. Takes the L2 norm of that gradient.
        3. Updates the EMA of that norm.
        4. Rescales the loss by ``weight / ema_norm`` so that all terms
           have comparable gradient magnitude after the backward pass.

        .. note::
           The key insight: without balancing, whichever loss has the
           largest gradient magnitude effectively dominates the shared
           encoder/decoder update, drowning out other objectives.
           Normalising by gradient norms ensures each loss contributes
           proportionally to its assigned weight.

        Args:
            losses: Mapping from loss name to scalar loss tensor.  Every
                name must appear in *self.weights*.
            shared_output: The intermediate tensor through which all losses
                flow (e.g. the encoder output or quantiser output).  Must
                have ``requires_grad=True``.

        Returns:
            A single scalar tensor suitable for ``.backward()``.

        Raises:
            KeyError: If a loss name is not present in *self.weights*.
        """
        total = torch.tensor(0.0, device=shared_output.device)

        for name, loss in losses.items():
            if name not in self.weights:
                raise KeyError(f"Loss {name!r} not found in balancer weights")

            # Compute per-loss gradient w.r.t. the shared representation.
            (grad,) = torch.autograd.grad(
                loss,
                shared_output,
                retain_graph=True,
                create_graph=False,
            )
            grad_norm = grad.detach().norm().item()

            # Update EMA; on first encounter, seed with current norm.
            if name not in self._ema_norms:
                self._ema_norms[name] = grad_norm
            else:
                d = self.ema_decay
                self._ema_norms[name] = d * self._ema_norms[name] + (1.0 - d) * grad_norm

            ema = self._ema_norms[name]
            # Avoid division by zero when a loss is effectively constant.
            scale = self.weights[name] / max(ema, 1e-8)

            total = total + scale * loss

        return total
