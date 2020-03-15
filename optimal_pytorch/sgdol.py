"""
Created on Mar 14th, 2020

@author Zhenxun Zhuang.
"""

from .optimizer import Optimizer

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

    def __init__(self, params, smoothness=10.0, alpha=10.0, weight_decay=0):
        if smoothness < 0.0:
            raise ValueError("Invalid smoothness value: {}".format(smoothness))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay)
        super(SGDOL, self).__init__(params, defaults)

        self._alpha = alpha
        self._smoothness = smoothness
        
        # Indicate whether we have obtained two stochastic gradients.
        self._is_first_grad = True 
        
        # Initialization.
        self._sum_inner_prods = alpha
        self._sum_grad_normsq = alpha
        self._lr = 1.0 / smoothness

    def __setstate__(self, state):
        super(SGDOL, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self._is_first_grad:
            # If it is the first mini-batch, just save the gradient for later
            # use and continue.
            for group in self.param_groups:
                weight_decay = group['weight_decay']

                for p in group['params']:                    
                    state = self.state[p]

                    if p.grad is None:
                        state['first_grad'] = None
                        continue

                    first_grad = p.grad.data
                    if weight_decay != 0:
                        first_grad.add_(weight_decay, p.data)
                    if first_grad.is_sparse:
                        raise RuntimeError(
                            'SGDOL does not support sparse gradients')

                    state['first_grad'] = first_grad.clone()
        else:
            for group in self.param_groups:
                weight_decay = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    second_grad = p.grad.data
                    if weight_decay != 0:
                        second_grad.add_(weight_decay, p.data)
                    if second_grad.is_sparse:
                        raise RuntimeError(
                            'SGDOL does not support sparse gradients')

                    state = self.state[p]
                    if state['first_grad'] is None:
                        continue

                    first_grad = state['first_grad']
                    
                    # Accumulate ||g_t||^2_2.
                    first_grad_norm = first_grad.norm()
                    first_grad_normsq = first_grad_norm * first_grad_norm
                    self._sum_grad_normsq += float(first_grad_normsq)
                    
                    # Accumulate <g_t, g'_t>.
                    cip = second_grad.view(-1).dot(first_grad.view(-1))
                    self._sum_inner_prods += float(cip)

            # Compute the step-size of the next round.
            lr = self._lr
            smoothness = self._smoothness
            lr_next = self._sum_inner_prods / (smoothness * self._sum_grad_normsq)
            lr_next = max(min(lr_next, 2.0/smoothness), 0.0)
            self._lr = lr_next

            # Update the parameters.
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if state['first_grad'] is None:
                        continue

                    first_grad = state['first_grad']

                    p.data.add_(-lr, first_grad)

        self._is_first_grad = not self._is_first_grad

        return loss
