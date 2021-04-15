import pytest
import torch
import optimal_pytorch as optim
from optimal_pytorch import Cocob, Recursive, ONSBet, Regralizer


def synthetic(x: torch.Tensor) -> torch.Tensor:
    """
    Synthetic function defined in https://arxiv.org/pdf/1912.01823.pdf
    """
    sample = torch.rand(1, dtype=torch.float)
    if sample < 0.002:
        loss = 999 * (x**2) / 2
    else:
        loss = -x
    return loss


def sinusoidal(x: torch.Tensor) -> torch.Tensor:
    """
    Function defined in http://infinity77.net/global_optimization/test_functions_1d.html
    f(x) = -x * sin(x), for x in [0, 10].
    """
    return -x * torch.sin(x)


cases = [
    (synthetic, 0., 0.5),
    (sinusoidal, 10., 7.9787),
]


optimizers = [
    Cocob,
    Recursive,
    ONSBet,
    Regralizer
]


@pytest.mark.parametrize('case', cases, ids=lambda x: f'{x[0].__name__} {x[1:]}')
@pytest.mark.parametrize('optimizer', optimizers, ids=lambda x: f'{x.__name__}')
def test_benchmark_function(case, optimizer):
    func, initial_state, min_loc = case
    optimizer_class = optimizer
    x = torch.tensor(initial_state).requires_grad_(True)
    x_min = torch.tensor(min_loc)
    opt = optimizer_class([x])
    iterations = 50
    model_history = torch.zeros(iterations)
    for i in range(iterations):
        opt.zero_grad()
        f = func(x)
        f.backward(retain_graph=True, create_graph=True)
        opt.step()
        model_history[i] = x.clone().detach()
    # parameter-free methods in 1d go exponentiall fast
    # towards the minimum and then jump back
    assert any(torch.abs(x_min - model_history) <= .01)
