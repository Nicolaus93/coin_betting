from typing import TYPE_CHECKING, Any, Optional, Callable
import torch
from torch.optim import Optimizer
from torch import norm

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ("Regralizer",)


class Regralizer(Optimizer):
    r"""Implements FTRL with rescaled gradients and linearithmic regularizer.

    It has been proposed in "Parameter-free Stochastic Optimization of
    Variationally Coherent Functions", https://arxiv.org/pdf/2102.00236.pdf

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts
            defining parameter groups.
        lr (float): Learning rate (default: 1e-1).
    """

    def __init__(self, params: _params_t, lr: float = 0.1):
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")

        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        for group in self.param_groups:  # state initialization
            for p in group["params"]:
                state = self.state[p]
                state["x0"] = p.clone().detach()
                state["theta"] = torch.zeros_like(p)
                state["S2"] = 4
                state["Q"] = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    msg = "Regralizer does not support sparse gradients!"
                    raise RuntimeError(msg)

                state = self.state[p]

                # retrieve params
                x0 = state["x0"]
                theta = state["theta"]
                S2 = state["S2"]
                Q = state["Q"]

                # update
                ell_t_squared = norm(grad * lr) ** 2
                theta.add_(grad * lr, alpha=-1)
                S2 += ell_t_squared
                Q += ell_t_squared / S2
                theta_norm = norm(theta)
                if theta_norm <= S2:
                    p.data = x0 + theta / (2 * S2) * torch.exp(
                        theta_norm ** 2 / (4 * S2) - Q
                    )
                else:
                    p.data = x0 + theta / (2 * theta_norm) * torch.exp(
                        theta_norm / 2 - S2 / 4 - Q
                    )

                # store params
                state["S2"] = S2
                state["Q"] = Q

        return loss
