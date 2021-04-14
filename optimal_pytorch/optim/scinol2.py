import torch
from torch.optim import Optimizer
from typing import TYPE_CHECKING, Any, Optional, Callable

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ('Scinol2',)


class Scinol2(Optimizer):
    """
    Implements SCale INvariant ONline Learning 2 (SCINOL2) algorithm.

    Proposed in "Adaptive Scale-Invariant Online Algorithms for Learning Linear Models",
    https://arxiv.org/pdf/1902.07528.pdf

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        eps (float): Regret at 0 of outer optimizer (initial wealth).
    """

    def __init__(self, params: _params_t, eps: float = 1.):
        if not 0.0 < eps:
            raise ValueError(f"Invalid eps value: {eps}")

        defaults = dict(eps=eps)
        super(Scinol2, self).__init__(params, defaults)
        for group in self.param_groups:  # State initialization
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                state["S_square"] = torch.zeros_like(p)
                state["G"] = torch.zeros_like(p)
                state["M"] = torch.full_like(p, 1e-8)
                state["eta"] = torch.full_like(p, eps)

    def update(self, xt: torch.Tensor) -> None:

        for group in self.param_groups:
            for p in group["params"]:

                state = self.state[p]
                M = state["M"]
                G = state["G"]
                S_square = state["S_square"]
                eta = state["eta"]

                # update weights
                torch.max(M, torch.abs(xt).detach(), out=M)
                helper = torch.sqrt(S_square + torch.square(M))
                theta = G / helper
                p.data = eta * torch.sign(theta) * torch.min(
                    torch.abs(theta), torch.ones_like(theta)
                ) / (2 * helper)

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
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # retrieve parameters
                state = self.state[p]
                G = state["G"]
                S_square = state["S_square"]
                eta = state["eta"]

                # update
                G.add_(grad, alpha=-1)
                S_square.add_(torch.square(grad))
                eta.add_(grad * p, alpha=-1)

        return loss
