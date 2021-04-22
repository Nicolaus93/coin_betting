import pytest
import torch
import optimal_pytorch.coin_betting.torch as cb


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
    cb.Cocob,
    cb.Recursive,
    cb.ONSBet,
    cb.Regralizer,
    # cb.SGDOL,
    # cb.Scinol2
]


@pytest.mark.parametrize('optimizer', optimizers, ids=lambda x: f'{x.__name__}')
def test_step(optimizer):

    weight = torch.tensor([2., 2.]).requires_grad_()
    bias = torch.tensor(1.).requires_grad_()
    input = torch.tensor([1., 1.])

    opt = optimizer([weight, bias])
    loss = (weight @ input + bias).pow(2).sum()
    initial_value = loss.item()
    loss.backward()
    opt.step()
    loss = (weight @ input + bias).pow(2).sum()
    assert loss.item() < initial_value


not_sparse = [
    cb.Cocob,
    cb.ONSBet,
    cb.Recursive,
    cb.Regralizer,
    cb.Scinol2,
    cb.SGDOL
]

@pytest.mark.parametrize('optimizer', not_sparse, ids=lambda x: f'{x.__name__}')
def test_sparse_not_supported(optimizer):
    param = torch.randn(1, 1).requires_grad_(True)
    grad = torch.randn(1, 1).to_sparse()
    param.grad = grad
    opt = optimizer([param])
    opt.zero_grad()
    with pytest.raises(RuntimeError) as ctx:
        opt.step()
    msg = 'does not support sparse gradients!'
    assert msg in str(ctx.value)
