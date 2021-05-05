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
        eps (float): regret at 0 of outer optimizer (initial wealth).
        eps_v (float): regret at 0 of each coordinate of inner optimizer
            (per-coordinate initial wealth).
        inner (Callable): Inner optimizer, default set to Scinol2. Scinol2 corresponds
            to a scale-invariant online learning algorithm
            (https://arxiv.org/pdf/1902.07528.pdf).
        momentum (Bool): a boolean indicating whether to use the "momentum" analog
            given in the appendix of the paper (section D.2)
    """

    @staticmethod
    def betting_loss(coin: torch.Tensor, bet_fraction: torch.Tensor) -> torch.Tensor:
        r"""Loss function used for coin betting methods.

        Arguments:
            coin (torch.Tensor): outcome of the coin in [-1, 1]
            bet_fraction (torch.Tensor): fraction of the wealth used for the current bet.
        """
        return -torch.log(1 - coin.view(-1) @ bet_fraction.view(-1))

    def __init__(
        self,
        params: _params_t,
        eps: float = 1.,
        eps_v: float = 1.0,
        inner: Callable = Scinol2,
        momentum: float = 0.
    ):
        if eps < 0.0:
            raise ValueError("Invalid eps (outer wealth) value: {}".format(eps))
        if eps_v < 0.0:
            raise ValueError("Invalid eps_v (inner wealth) value: {}".format(eps_v))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(eps=eps, inner=inner, eps_v=eps_v, momentum=momentum)
        super().__init__(params, defaults)

        for group in self.param_groups:  # State initialization
            inner = group["inner"]
            momentum = group["momentum"]

            for p in group["params"]:
                state = self.state[p]
                state["initial_value"] = p.clone().detach()
                state["betting_fraction"] = torch.zeros_like(p, memory_format=torch.preserve_format).requires_grad_()
                state["inner_opt"] = inner([state["betting_fraction"]])  # NOTE: we create an inner optimizer for every group
                state["wealth"] = torch.tensor(eps)
                state["max_grad"] = torch.tensor(1e-8)
                if momentum > 0:
                    state["weighted_params_sum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["grad_norm_sum"] = 1

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # retrieve state
                state = self.state[p]
                betting_fraction = state["betting_fraction"]
                wealth = state["wealth"]
                inner_opt = state["inner_opt"]
                max_grad = state["max_grad"]
                x0 = state["initial_value"]

                # rescale gradients
                grad = p.grad
                if grad.is_sparse:
                    msg = "Recursive does not support sparse gradients!"
                    raise RuntimeError(msg)

                # In the paper it is assumed that ||g_t||_1 <= 1. To make this true we
                # need to divide by max_grad.
                # Furthermore, the gradients provided to INNER_OPT also have to satisfy
                # ||z_t||_\infty <= 1, hence we further divide the gradients by 2.
                # For more details see section 5 of the paper.
                grad_norm = torch.norm(grad, p=1).detach()
                torch.max(grad_norm, max_grad, out=max_grad)
                grad = grad / (2 * max_grad)

                # pass gradients to inner and update state
                wealth.add_(grad.view(-1) @ (p.view(-1) - x0.view(-1)), alpha=-1)
                if inner_opt.__class__.__name__ == "Scinol2":
                    inner_opt.update(grad)
                inner_opt.zero_grad()
                with torch.no_grad():
                    # we need to ensure that betting_fraction € [-0.5, 0.5]
                    # otherwise the wealth can become negative
                    # (see Thm. 1 assumption: ||vt||_\infty <= 1/2)
                    betting_fraction.clamp_(-0.5, 0.5)
                inner_loss = self.betting_loss(grad, betting_fraction)
                inner_loss.backward()
                inner_opt.step()
                p.data = x0 + wealth * betting_fraction

                if momentum > 0:
                    next_offset = wealth * betting_fraction
                    squared_grad_norm = grad_norm.pow(2).item()
                    state["grad_norm_sum"] = squared_grad_norm + momentum * state["grad_norm_sum"]
                    avg_offset = squared_grad_norm * (next_offset - state["weighted_params_sum"]) / state["grad_norm_sum"]
                    state["weighted_params_sum"] = avg_offset
                    p.data.add_(avg_offset)

        return loss
