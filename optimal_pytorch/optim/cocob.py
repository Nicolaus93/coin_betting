import torch
from torch.optim.optimizer import Optimizer
from .types import OptFloat, OptLossClosure, Params

__all__ = ('Cocob',)


class Cocob(Optimizer):
    """
    COntinuos COin Betting (COCOB) optimizer from the paper
    Training Deep Networks without Learning Rates Through Coin Betting.
    https://arxiv.org/abs/1705.07795
    """

    def __init__(
        self,
        params: Params,
        alpha: float = 100,
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay)
        self._alpha = alpha
        self._eps = eps

        super(Cocob, self).__init__(params, defaults)
    
    def grid_search_params(self):
        ranges = {}
        ranges['alpha'] = [100, 'use']
        ranges['weight_decay'] = [1e-3, 10, 'gen', 6]
        return ranges

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Cocob does not support sparse gradients, '
                        'please consider SparseAdam instead.'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # sum of the gradients
                    state['gradients_sum'] = torch.zeros_like(p)
                    # sum of the absolute values of the subgradients
                    state['grad_norm_sum'] = torch.zeros_like(p)
                    # Update the maximum observed scale
                    state['L'] = self._eps * torch.ones_like(p)
                    # tilde_w_t = w_1 - w_t
                    state['tilde_w'] = torch.zeros_like(p)
                    # reward
                    state['reward'] = torch.zeros_like(p)

                gradients_sum, grad_norm_sum, L, tilde_w, reward = (
                    state['gradients_sum'],
                    state['grad_norm_sum'],
                    state['L'],
                    state['tilde_w'],
                    state['reward'],
                )

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha = group['weight_decay'])

                # absolute value of current gradient vector
                abs_grad = torch.abs(grad)
                # update parameters
                L = torch.max(L, abs_grad)
                gradients_sum.add_(grad)
                grad_norm_sum.add_(abs_grad)
                reward = torch.max(reward - grad * tilde_w, torch.zeros_like(reward))
                den = L * torch.max(grad_norm_sum + L, self._alpha * L)
                x = (gradients_sum / den) * (reward + L)
                p.data.add_(-tilde_w - x)
                tilde_w = -x

                # state update
                state['gradients_sum'] = gradients_sum
                state['grad_norm_sum'] = grad_norm_sum
                state['L'] = L
                state['tilde_w'] = tilde_w
                state['reward'] = reward

        return loss
