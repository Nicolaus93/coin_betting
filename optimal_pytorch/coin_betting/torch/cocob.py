from typing import TYPE_CHECKING, Any, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer


if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ("Cocob",)


class Cocob(Optimizer):
    """
    Implements COntinuos COin Betting (COCOB) algorithm.

    Proposed in "Training Deep Networks without Learning Rates Through Coin Betting",
    https://arxiv.org/abs/1705.07795

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
                            parameter groups.
        alpha (float):     (hyper)parameter to guarantee that the fraction to bet
                            is at least alpha * L_{t, i} for each weight
                            (default: 100.0).
    """

    def __init__(self, params: _params_t, alpha: float = 100.0):
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(alpha=alpha)
        super().__init__(params, defaults)

        for group in self.param_groups:  # State initialization
            for p in group["params"]:
                state = self.state[p]
                state["w1"] = p.clone().detach()  # initial model
                state["grad_sum"] = torch.zeros_like(p)  # sum of the gradients
                state["abs_grad_sum"] = torch.zeros_like(
                    p
                )  # sum of abs value of gradients
                state["max_scale"] = torch.full_like(p, 1e-8)  # maximum observed scale
                state["reward"] = torch.zeros_like(p)  # cumulative reward

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates
                the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group["alpha"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = -p.grad.data  # negative gradient
                if grad.is_sparse:
                    msg = "Cocob does not support sparse gradients!"
                    raise RuntimeError(msg)

                # Retrieve parameters
                state = self.state[p]
                w1 = state["w1"]
                grad_sum = state["grad_sum"]
                abs_grad_sum = state["abs_grad_sum"]
                max_scale = state["max_scale"]
                reward = state["reward"]

                # Update parameters (inplace)
                abs_grad = torch.abs(grad)
                torch.max(max_scale, abs_grad, out=max_scale)
                grad_sum.add_(grad)
                abs_grad_sum.add_(abs_grad)
                torch.max(
                    reward + (p.data - w1) * grad, torch.zeros_like(reward), out=reward
                )
                p.data = w1 + grad_sum / (
                    max_scale * (torch.max(abs_grad_sum + max_scale, alpha * max_scale))
                ) * (reward + max_scale)

        return loss
