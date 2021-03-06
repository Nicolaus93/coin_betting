from typing import TYPE_CHECKING, Any, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer
from .scinol2 import Scinol2

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ("Recursive",)


class Recursive(Optimizer):
    r"""Implements Recursive optimizer.

    It has been proposed in "Matrix-Free Preconditioning in Online Learning",
    https://arxiv.org/pdf/1905.12721v1.pdf

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
                            parameter groups.
        eps (float):       Regret at 0 of outer optimizer (initial wealth).
        eps_v (float):     Regret at 0 of each coordinate of inner optimizer
                            (per-coordinate initial wealth).
        inner (Callable):  Inner optimizer, default set to Scinol2. Scinol2 corresponds
                            to a scale-invariant online learning algorithm
                            (https://arxiv.org/pdf/1902.07528.pdf).
        TODO: add momentum.
    """

    @staticmethod
    def betting_loss(coin: torch.Tensor, bet_fraction: torch.Tensor) -> torch.Tensor:
        r"""Loss function used for coin betting methods.

        Arguments:
            coin: outcome of the coin in [-1, 1]
            bet_fraction: fraction of the wealth used for the current bet.
        """
        return -torch.log(1 - coin.view(-1) @ bet_fraction.view(-1))

    def __init__(
        self,
        params: _params_t,
        eps: float = 1.0,
        eps_v: float = 1.0,
        inner: Callable = Scinol2,
    ):
        if eps < 0.0:
            raise ValueError("Invalid eps (outer wealth) value: {}".format(eps))
        if eps_v < 0.0:
            raise ValueError("Invalid eps_v (inner wealth) value: {}".format(eps_v))

        defaults = dict(eps=eps, inner=inner)
        super().__init__(params, defaults)
        for group in self.param_groups:  # State initialization
            inner = group["inner"]
            for p in group["params"]:
                state = self.state[p]
                state["vt"] = p.clone().detach().requires_grad_(True)
                state["inner"] = inner([state["vt"]])
                state["wealth"] = torch.tensor(eps)
                state["max_grad"] = torch.tensor(1e-8)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue

                # retrieve state
                state = self.state[p]
                vt = state["vt"]
                wealth = state["wealth"]
                inner = state["inner"]
                max_grad = state["max_grad"]

                # rescale gradients
                grad = p.grad
                if grad.is_sparse:
                    msg = "Recursive does not support sparse gradients!"
                    raise RuntimeError(msg)

                torch.max(torch.norm(grad, p=1).detach(), max_grad, out=max_grad)
                grad = grad / (2 * max_grad)

                # pass gradients to inner and update state
                wealth.add_(grad.view(-1) @ p.view(-1), alpha=-1)
                if inner.__class__.__name__ == "Scinol2":
                    inner.update(grad)
                inner.zero_grad()
                with torch.no_grad():
                    # we need to ensure that vt ??? [-0.5, 0.5]
                    # otherwise the wealth can become negative
                    vt.clamp_(-0.5, 0.5)
                inner_loss = self.betting_loss(grad, vt)
                inner_loss.backward()
                inner.step()
                p.data = wealth * vt

        return loss
