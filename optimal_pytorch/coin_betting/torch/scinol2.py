from typing import TYPE_CHECKING, Any, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ("Scinol2",)


class Scinol2(Optimizer):
    r"""Implements SCale INvariant ONline Learning 2 (SCINOL2) algorithm.

    Proposed in "Adaptive Scale-Invariant Online Algorithms for Learning Linear Models",
    https://arxiv.org/pdf/1902.07528.pdf

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        eps (float): Regret at 0 of outer optimizer (initial wealth).
    """

    def __init__(self, params: _params_t, eps: float = 1.0):
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")

        defaults = dict(eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:  # State initialization
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                state["S_square"] = torch.zeros_like(p)
                state["G"] = torch.zeros_like(p)
                state["M"] = torch.full_like(p, 1e-8)
                state["eta"] = torch.full_like(p, eps)

    def update(self, xt: torch.Tensor) -> None:
        r"""
        Scinol2 is an improper algorithm, meaning it has access to the next feature
        vector before providing a prediction in output. It then uses this information
        to update its state.
        Arguments:
            xt: feature vector for the "next" round.
        """
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
                p.data = (
                    eta
                    * torch.sign(theta)
                    * torch.min(torch.abs(theta), torch.ones_like(theta))
                    / (2 * helper)
                )

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
                if grad.is_sparse:
                    msg = "Scinol2 does not support sparse gradients!"
                    raise RuntimeError(msg)

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
