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
        self.received_xt = False
        for group in self.param_groups:  # State initialization
            # there should only be a linear layer (aka 1 group)
            eps = group["eps"]
            for p in group["params"]:
                state = self.state[p]
                state["initial_value"] = p.clone().detach()
                state["squared_grad_sum"] = torch.zeros_like(p)
                state["grad_sum"] = torch.zeros_like(p)
                state["max_scale"] = torch.full_like(p, 1e-8)
                state["eta"] = torch.full_like(p, eps)

    def observe(self, next_feat_vector: torch.Tensor) -> None:
        r"""
        Scinol2 is an improper algorithm, meaning it has access to the next feature
        vector before providing a prediction in output. It then uses this information
        to update its state.
        Arguments:
            next_features: feature vector for the "next" round.
        """
        self.received_xt = True
        for group in self.param_groups:
            # NOTE: we shouldn't have multiple groups if the model
            # is LINEAR unless we compose multiple linear Models
            # (which is anyway equivalent to a single linear model)

            for i, p in enumerate(group["params"]):
                # here we assume that model parameters should be
                # equal to [weights, bias], with weights coming first
                xt = torch.tensor(1.) if i > 0 else next_feat_vector

                state = self.state[p]
                x0 = state["initial_value"]
                max_scale = state["max_scale"]
                grad_sum = state["grad_sum"]
                squared_grad_sum = state["squared_grad_sum"]
                eta = state["eta"]

                # update weights
                torch.max(
                    max_scale, torch.abs(xt).detach(), out=max_scale
                )
                helper = torch.sqrt(squared_grad_sum + torch.square(max_scale))
                theta = grad_sum / helper
                p.data = x0 + (
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

                if not self.received_xt:
                    msg = "Scinol2 has not received the next feature vector!"
                    raise RuntimeError(msg)

                # update parameters
                state = self.state[p]
                x0 = state["initial_value"]
                state["grad_sum"].add_(grad, alpha=-1)
                state["squared_grad_sum"].add_(grad.pow(2))
                state["eta"].add_(grad * (p - x0), alpha=-1)

        self.received_xt = False
        return loss
