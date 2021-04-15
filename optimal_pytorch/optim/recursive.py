import torch
from torch.optim import Optimizer
from typing import TYPE_CHECKING, Any, Optional, Callable
from scinol2 import Scinol2

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ('Recursive',)


class Recursive(Optimizer):
    """
    Implements Recursive optimizer.

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
    def betting_loss(gt: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
        return -torch.log(1 - gt.view(-1) @ vt.view(-1))

    def __init__(self, params: _params_t, eps: float = 1., eps_v: float = 1.,
                 inner: Optimizer = Scinol2):
        defaults = dict(eps=eps, inner=inner)
        super(Recursive, self).__init__(params, defaults)
        for group in self.param_groups:  # State initialization
            inner = group['inner']
            for p in group['params']:
                state = self.state[p]
                state['vt'] = p.clone().detach().requires_grad_(True)
                state['inner'] = inner([state['vt']])
                state['wealth'] = torch.tensor(eps)
                state['max_grad'] = torch.tensor(1e-8)

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
                torch.max(torch.norm(grad, p=1).detach(), max_grad, out=max_grad)
                grad = grad / (2 * max_grad)

                # pass gradients to inner and update state
                wealth.add_(grad.view(-1) @ p.view(-1), alpha=-1)
                if inner.__class__.__name__ == 'Scinol2':
                    inner.update(grad)
                inner.zero_grad()
                with torch.no_grad():
                    # we need to ensure that vt € [-0.5, 0.5]
                    # otherwise the wealth can become negative
                    vt.clamp_(-.5, .5)
                inner_loss = self.betting_loss(grad, vt)
                inner_loss.backward()
                inner.step()
                p.data = wealth * vt

        return loss
