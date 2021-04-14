import torch
from torch.optim import Optimizer
from typing import TYPE_CHECKING, Any, Optional, Callable

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


__all__ = ('Recursive',)


class ONSBet(Optimizer):
    """
    Implements Diagonal Betting optimizer.

    It has been proposed in "Matrix-Free Preconditioning in Online Learning",
    https://arxiv.org/pdf/1905.12721v1.pdf and https://arxiv.org/pdf/1802.06293.pdf

    Args:
        eps (float): Initial wealth of the algorithm.

    TODO: - Should we remove vt from state? It can be (re)computed in every step.
          This would save memory, but require more computations.
          - Add momentum.
    """

    def __init__(self, params: _params_t, eps: float = 1.):
        defaults = dict(eps=eps)
        super(ONSBet, self).__init__(params, defaults)
        for group in self.param_groups:  # state initialization
            for p in group["params"]:
                state = self.state[p]
                state["wealth"] = torch.full_like(p, eps)
                state["At"] = torch.full_like(p, 5)
                state["vt"] = p.clone().detach().clamp(-.5, .5)
                state["zt_sum"] = torch.zeros_like(p)
                state["max_grad"] = torch.full_like(p, 1e-8)

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
                max_grad = state["max_grad"]
                wealth = state["wealth"]
                At = state["At"]
                zt_sum = state["zt_sum"]
                vt = state["vt"]

                # compute gradients
                grad = p.grad
                xt = vt * wealth               # retrieve old x_t
                grad[grad * (xt - p) < 0] = 0  # \tilde{g}_t

                # update state
                wealth.add_(-xt * grad)
                zt = grad / (1 - grad * vt)
                zt_sum.add_(zt)
                At.add_(torch.square(zt))
                vt = torch.clamp(-zt_sum / At, -.5, .5)
                p.data = torch.clamp(vt * wealth, -.5, .5)
                # clip the inner betting fraction if At < 1 (see appendix D.4)
                p.data[At < 1] = torch.clamp(.1 * wealth[At < 1], -.5, .5)

                # store
                state["vt"] = vt

        return loss


class Recursive(Optimizer):
    """
    Implements Recursive optimizer.

    It has been proposed in "Matrix-Free Preconditioning in Online Learning",
    https://arxiv.org/pdf/1905.12721v1.pdf

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        eps (float): Regret at 0 of outer optimizer (initial wealth).
        eps_v (float): Regret at 0 of each coordinate of inner optimizer
            (per-coordinate initial wealth).
        inner (Callable): Inner optimizer. ONSBet corresponds to using coin-betting
            reduction with ONS as base optimizer. Scinol corresponds to scale-invariant
            online learning algorithm.
    """

    @staticmethod
    def betting_loss(gt, vt):
        return -torch.log(1 - gt.view(-1) @ vt.view(-1))

    def __init__(
        self,
        params: _params_t,
        eps: float = 1.,
        eps_v: float = 1.,
        inner: Optimizer = ONSBet,
    ):
        defaults = dict(eps=eps, inner=inner)
        super(Recursive, self).__init__(params, defaults)
        for group in self.param_groups:  # State initialization
            inner = group["inner"]
            for p in group["params"]:
                state = self.state[p]
                state["vt"] = p.clone().detach().requires_grad_(True)
                state["inner"] = inner([state["vt"]])
                state["wealth"] = torch.tensor(eps)
                state["max_grad"] = torch.tensor(1e-8)

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
                if inner == Scinol:
                    inner.step(grad)
                    inner.zero_grad()
                    inner_loss = self.betting_loss(grad, vt)
                    inner_loss.backward()
                else:
                    inner.zero_grad()
                    inner_loss = self.betting_loss(grad, vt)
                    inner_loss.backward()
                    inner.step()
                p.data = wealth * vt

        return loss
