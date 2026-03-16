"""Vendored Muon optimizer — Newton-Schulz orthogonalized SGD momentum.

This module provides a self-contained implementation of the Muon optimizer,
based on Keller Jordan's reference code. Muon replaces each SGD-momentum
update with the nearest semi-orthogonal matrix via Newton-Schulz iteration,
amplifying under-represented gradient directions while suppressing dominant
ones.

Usage::

    from train.utils.muon import Muon

    # Use Muon for 2-D hidden-layer weights only; route embeddings,
    # norms, and biases to AdamW separately.
    optimizer = Muon(hidden_params, lr=0.02, momentum=0.95)

Dependencies:
    torch

License: MIT

Citation::

    @misc{jordan2024muon,
      title   = {Muon: An optimizer for hidden layers in neural networks},
      author  = {Keller Jordan and Yuchen Jin and Vlado Boza
                 and Jiacheng You and Franz Cesista and Laker Newhouse
                 and Jeremy Bernstein},
      year    = {2024},
      url     = {https://kellerjordan.github.io/muon/}
    }
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the nearest orthogonal matrix via Newton-Schulz iteration.

    Applies 5 (by default) iterations of a quintic Newton-Schulz polynomial
    to drive ``G`` toward the closest semi-orthogonal matrix.  The three
    polynomial coefficients were numerically tuned to maximise convergence
    speed for matrices whose spectral norm is ≤ 1.

    Args:
        G: A 2-D tensor (or batched 2-D) to orthogonalize.
        steps: Number of Newton-Schulz iterations.
        eps: Small constant added to the spectral-norm estimate to prevent
            division by zero.

    Returns:
        The orthogonalized tensor in bfloat16.

    Example::

        >>> G = torch.randn(128, 64)
        >>> Q = newtonschulz5(G, steps=5)
        >>> Q.shape
        torch.Size([128, 64])
    """
    assert G.ndim >= 2, "newtonschulz5 requires a 2-D (or higher) tensor"

    # Tuned quintic coefficients for spectral-norm ≤ 1 convergence.
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    # Normalise so spectral norm ≈ 1; this is the convergence prerequisite.
    X = X / (X.norm() + eps)

    if G.size(0) > G.size(1):
        # Tall matrix: iterate on X^T X (cheaper).
        for _ in range(steps):
            A = X.T @ X
            B = b * A + c * A @ A  # quintic polynomial inner term
            X = a * X + X @ B
    else:
        # Wide or square matrix: iterate on X X^T.
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

    return X


class Muon(Optimizer):
    """SGD with Nesterov momentum and Newton-Schulz orthogonalization.

    All parameters **must** be 2-D or higher.  4-D convolutional weights are
    flattened to 2-D before the Newton-Schulz step, then reshaped back.

    Args:
        params: Iterable of parameters (must have ``ndim >= 2``).
        lr: Learning rate.  Typical range is ``[0.01, 0.1]``, i.e. SGD-scale.
        momentum: Momentum coefficient for the SGD update.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations per step.

    Raises:
        ValueError: If ``lr``, ``momentum``, or ``ns_steps`` are invalid.

    Example::

        >>> model = torch.nn.Linear(256, 128)
        >>> opt = Muon(model.parameters(), lr=0.02)
        >>> opt.step()
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:  # noqa: D401
        """Perform a single optimisation step.

        Args:
            closure: An optional closure that re-evaluates the model and
                returns the loss.

        Returns:
            The loss value if *closure* was supplied, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim >= 2, "Muon only supports parameters with ndim >= 2"

                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf

                # Flatten ≥3-D params (e.g. Conv2d) to 2-D for orthogonalization.
                orig_shape = update.shape
                if update.ndim > 2:
                    update = update.flatten(1)

                update = newtonschulz5(update, steps=ns_steps)
                update = update.view(orig_shape).to(p.dtype)

                p.add_(update, alpha=-lr)

        return loss
