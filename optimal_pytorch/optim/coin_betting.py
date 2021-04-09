import torch
from torch.optim import Optimizer
from typing import TYPE_CHECKING, Any, Optional, Callable

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ('CoinBetting',)


class CoinBetting(Optimizer):
    """
    COntinuos COin Betting (COCOB) optimizer from the paper
    Training Deep Networks without Learning Rates Through Coin Betting.
    https://arxiv.org/abs/1705.07795
    """

    def __init__(self, params: _params_t, alpha: float = 100.0, eps: float = 1e-8):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(alpha=alpha, eps=eps)
        super(Cocob, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = -p.grad.data  # negative gradient
                if grad.is_sparse:
                    msg = (
                        'Cocob does not support sparse gradients, '
                        'please consider SparseAdam instead.'
                    )
                    raise RuntimeError(msg)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['w1'] = p.clone().detach()
                    # sum of the gradients
                    state['theta'] = torch.zeros_like(p).detach()
                    # sum of the absolute values of the subgradients
                    state['Gt'] = torch.zeros_like(p).detach()
                    # maximum observed scale
                    state['Lt'] = eps * torch.ones_like(p).detach()
                    # cumulative reward
                    state['reward'] = torch.zeros_like(p).detach()

                # Retrieve parameters
                w1 = state['w1']
                theta = state['theta']
                Gt = state['Gt']
                Lt = state['Lt']
                reward = state['reward']

                abs_grad = torch.abs(grad)
                # Update
                Lt = torch.max(Lt, abs_grad)
                theta.add_(grad)
                Gt.add_(abs_grad)
                reward = torch.max(reward + (p.data - w1) * grad, torch.zeros_like(reward))
                p.data = w1 + theta / (Lt * (torch.max(Gt + Lt, alpha * Lt))) * (reward + Lt)

        return loss
