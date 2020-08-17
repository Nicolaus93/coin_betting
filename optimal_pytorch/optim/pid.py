import torch
from torch.optim.optimizer import Optimizer

from .types import OptFloat, OptLossClosure, Params


class PID(Optimizer):
    r"""Implements PID optimization algorithm.
    It has been proposed in `A PID Controller Approach for Stochastic
    Optimization of Deep Networks`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.0)
        weight_decay: weight decay (L2 penalty) (default: 0.0)
        dampening: dampening for momentum (default: 0.0)
        derivative: D part of the PID (default: 10.0)
        integral: I part of the PID (default: 5.0)
    Example:
        >>> import optimal_pytorch as optim
        >>> optimizer = optim.PID(model.parameters(), lr=0.001, momentum=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0,
        weight_decay: float = 0.0,
        integral: float = 5.0,
        derivative: float = 10.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            integral=integral,
            derivative=derivative,
        )
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        super(PID, self).__init__(params, defaults)

    def grid_search_params(self):
        ranges = {}
        ranges['lr'] = [1e-5, 1, 'gen', 6]
        ranges['momentum'] = [0.1, 0.5, 0.9, 0.99, 'use']
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            integral = group['integral']
            derivative = group['derivative']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'i_buffer' not in param_state:
                        i_buf = param_state['i_buffer'] = torch.zeros_like(p)
                        i_buf.mul_(momentum).add_(d_p)
                    else:
                        i_buf = param_state['i_buffer']
                        i_buf.mul_(momentum).add_(d_p, alpha = 1 - dampening)
                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(
                            p
                        )
                        g_buf = d_p

                        d_buf = param_state['d_buffer'] = torch.zeros_like(p)
                        d_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        d_buf = param_state['d_buffer']
                        g_buf = param_state['grad_buffer']
                        d_buf.mul_(momentum).add_(d_p - g_buf, alpha = 1 - momentum)
                        self.state[p]['grad_buffer'] = d_p.clone()

                    d_p = d_p.add_(integral, i_buf).add_(d_buf, alpha = derivative)
                p.data.add_(-group['lr'], d_p)
        return loss
