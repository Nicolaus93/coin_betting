import pytest
import torch
import optimal_pytorch.coin_betting.torch as cb
from functools import partial


@pytest.fixture(autouse=True)
def set_torch_seed():
    torch.manual_seed(42)
    yield


cb_opt = [
    cb.Recursive,
    cb.ONSBet,
    cb.Scinol2
]


@pytest.mark.parametrize('optimizer', cb_opt, ids=lambda x: f'{x.__name__}')
def test_invalid_eps(optimizer):
    weight = torch.randn(10, 5).float().requires_grad_()
    bias = torch.randn(10).float().requires_grad_()
    with pytest.raises(ValueError):
        optimizer([weight, bias], eps=-1)



optimizers = [
    (cb.Cocob, 'Cocob'),
    (partial(cb.Recursive, inner=cb.Cocob), 'Recursive'),
    (cb.ONSBet, 'ONSBet'),
    (partial(cb.Regralizer, lr=.1), 'Regralizer'),
    (cb.SGDOL, 'SGDOL'),
    (cb.Scinol2, 'Scinol2')
]


@pytest.mark.parametrize('optimizer', optimizers, ids=lambda x: x[1])
def test_step(optimizer):

    algo, name = optimizer
    weights = torch.tensor([2., 2.]).requires_grad_()
    bias = torch.tensor(1.).requires_grad_()
    features = torch.tensor([1., 1.])
    opt = algo([weights, bias])
    n = 3 if name == 'SGDOL' else 2
    loss_values = []
    for _ in range(n):
        if name == 'Scinol2':
            opt.observe(features)
        loss = (weights @ features + bias).pow(2)
        loss_values.append(loss.item())
        loss.backward()
        opt.step()

    assert loss_values[-1] < loss_values[-2]


not_sparse = [
    (cb.Cocob, 'Cocob'),
    (cb.ONSBet, 'ONSBet'),
    (cb.Recursive, 'Recursive'),
    (partial(cb.Regralizer, lr=.1), 'Regralizer'),
    (cb.Scinol2, 'Scinol2'),
    (cb.SGDOL, 'SGDOL')
]

@pytest.mark.parametrize('optimizer', not_sparse, ids=lambda x: x[1])
def test_sparse_not_supported(optimizer):

    algo, name = optimizer
    param = torch.randn(1, 1).requires_grad_(True)
    grad = torch.randn(1, 1).to_sparse()
    param.grad = grad
    opt = algo([param])
    opt.zero_grad()
    with pytest.raises(RuntimeError) as ctx:
        opt.step()
    msg = 'does not support sparse gradients!'
    assert msg in str(ctx.value)
