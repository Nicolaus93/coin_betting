from typing import TYPE_CHECKING, Any, Optional, Callable
from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class SGDOL(Optimizer):
    r"""Implement the SGDOL Algorithm.

    Description:
    This algorithm was proposed in "Surrogate Losses for Online Learning of
    Stepsizes in Stochastic Non-Convex Optimization" which can be checked out
    at: https://arxiv.org/abs/1901.09068

    The online learning algorithm used here is
    "Follow-The-Regularized-Leader-Proximal" as described in the paper.

    Arguments:
    - params (iterable): iterable of parameters to optimize or dicts
          defining parameter groups.
    - smoothness (float, optional): the assumed smoothness of the loss
          function (default: 10).
    - alpha (float, optional): the parameter alpha used in the inital
          regularizer (default: 10)
    - weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params: _params_t,
        smoothness: float = 10.0,
        alpha: float = 10.0,
        weight_decay: float = 0.0,
    ) -> None:
        if smoothness <= 0.0:
            raise ValueError(f"Invalid smoothness value: {smoothness}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(smoothness=smoothness, alpha=alpha, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:  # state initialization
            alpha = group["alpha"]
            smoothness = group["smoothness"]
            for p in group["params"]:
                state = self.state[p]
                state["sum_inner_prods"] = alpha
                state["sum_grad_normsq"] = alpha
                state["lr"] = 1.0 / smoothness
                state[
                    "is_first_grad"
                ] = True  # Indicate whether we have obtained two stochastic gradients.

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
            weight_decay = group["weight_decay"]
            smoothness = group["smoothness"]

            for p in group["params"]:
                state = self.state[p]

                if p.grad is None:
                    if state["is_first_grad"]:
                        state["first_grad"] = None
                        state["is_first_grad"] = False
                    # if it's second grad, just skip this round
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SGDOL does not support sparse gradients!")
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                if state["is_first_grad"]:
                    # If it is the first mini-batch, just save the gradient for later
                    # use and continue.
                    state["first_grad"] = grad.clone().detach()
                else:
                    first_grad = state["first_grad"]
                    if first_grad is None:
                        continue

                    # Accumulate ||g_t||^2_2.
                    first_grad_norm = first_grad.norm()
                    first_grad_normsq = first_grad_norm.pow(2)
                    state["sum_grad_normsq"] += first_grad_normsq.item()

                    # Accumulate <g_t, g'_t>.
                    cip = grad.view(-1) @ first_grad.view(-1)
                    state["sum_inner_prods"] += cip.item()

                    # Compute the step-size of the next round and update the parameters.
                    lr = state["sum_inner_prods"] / (
                        smoothness * state["sum_grad_normsq"]
                    )
                    lr = max(min(lr, 2.0 / smoothness), 0.0)
                    p.data.add_(first_grad, alpha=-lr)

                # update state
                state["is_first_grad"] = not state["is_first_grad"]

        return loss
