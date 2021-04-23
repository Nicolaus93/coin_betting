from typing import TYPE_CHECKING, Any, Optional, Callable
import torch
from torch.optim import Optimizer


if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ("ONSBet",)


class ONSBet(Optimizer):
    r"""Implements Diagonal Betting optimizer/Coin-Betting through ONS.

    It has been proposed in "Matrix-Free Preconditioning in Online Learning",
    https://arxiv.org/pdf/1905.12721v1.pdf and
    "Black-Box Reductions for Parameter-free Online Learning in Banach Spaces",
    https://arxiv.org/pdf/1802.06293.pdf

    Args:
        eps (float): Initial wealth of the algorithm.

    TODO: - Should we remove vt from state? It can be (re)computed in every step.
          This would save memory, but require more computations.
    """

    def __init__(self, params: _params_t, eps: float = 1.0):
        if eps < 0.0:
            raise ValueError("Invalid eps (initial wealth) value: {}".format(eps))

        defaults = dict(eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:  # state initialization
            for p in group["params"]:
                state = self.state[p]
                state["wealth"] = torch.full_like(p, eps)
                state["At"] = torch.full_like(p, 5)
                state["vt"] = p.clone().detach().clamp(-0.5, 0.5)
                state["zt_sum"] = torch.zeros_like(p)
                state["max_grad"] = torch.full_like(p, 1e-8)

    @torch.no_grad()
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
                max_grad = state["max_grad"]
                wealth = state["wealth"]
                At = state["At"]
                zt_sum = state["zt_sum"]
                vt = state["vt"]

                # compute gradients
                grad = p.grad
                if grad.is_sparse:
                    msg = "ONSBet does not support sparse gradients!"
                    raise RuntimeError(msg)

                torch.max(max_grad, torch.abs(grad), out=max_grad)
                grad /= max_grad
                xt = vt * wealth  # retrieve old x_t
                grad[grad * (xt - p) < 0] = 0  # \tilde{g}_t

                # update state
                wealth.add_(-xt * grad)
                zt = grad / (1 - grad * vt)
                zt_sum.add_(zt)
                At.add_(torch.square(zt))
                vt = torch.clamp(-zt_sum / At, -0.5, 0.5)
                p.data = torch.clamp(vt * wealth, -0.5, 0.5)
                # # clip the inner betting fraction if At < 1 (see appendix D.4)
                # check what's going on, this doesn't work!
                # p.data[At < 1] = torch.clamp(.1 * wealth[At < 1], -.5, .5)

                # store
                state["vt"] = vt

        return loss
