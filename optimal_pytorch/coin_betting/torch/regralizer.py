from typing import TYPE_CHECKING, Any, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

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

    def __init__(self, params: _params_t, lr: float, momentum: float = 0.0):
        if lr <= 0:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if momentum < 0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        for group in self.param_groups:  # state initialization
            momentum = group["momentum"]
            for p in group["params"]:
                state = self.state[p]
                state["initial_value"] = p.clone().detach()
                state["rescaled_grad_sum"] = torch.zeros_like(p)
                state["rescaled_grad_norm_sum"] = 4
                state["Q_sum"] = 0
                if momentum > 0:
                    state["weighted_params_sum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["grad_norm_sum"] = 1

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
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    msg = "Regralizer does not support sparse gradients!"
                    raise RuntimeError(msg)

                state = self.state[p]

                # retrieve params
                x0 = state["initial_value"]
                theta = state["rescaled_grad_sum"]
                S2_sum = state["rescaled_grad_norm_sum"]
                Q_sum = state["Q_sum"]

                # update
                ell_t_squared = (grad * lr).norm().pow(2)
                theta.add_(grad * lr, alpha=-1)
                S2_sum += ell_t_squared
                Q_sum += ell_t_squared / S2_sum
                theta_norm = theta.norm()
                if theta_norm <= S2_sum:
                    next_offset = (
                        theta
                        / (2 * S2_sum)
                        * torch.exp(theta_norm ** 2 / (4 * S2_sum) - Q_sum)
                    )
                else:
                    next_offset = (
                        theta
                        / (2 * theta_norm)
                        * torch.exp(theta_norm / 2 - S2_sum / 4 - Q_sum)
                    )
                p.data = x0 + next_offset

                if momentum > 0:
                    state["grad_norm_sum"] = (
                        ell_t_squared + momentum * state["grad_norm_sum"]
                    )
                    avg_offset = next_offset - state["weighted_params_sum"]
                    state["weighted_params_sum"] = avg_offset
                    p.data.add_(
                        avg_offset, alpha=ell_t_squared / state["grad_norm_sum"]
                    )

                # store params
                state["rescaled_grad_norm_sum"] = S2_sum
                state["Q_sum"] = Q_sum

        return loss
