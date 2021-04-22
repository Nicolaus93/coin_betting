import pytest
import torch
from optimal_pytorch.test_functions import Ackley, Sinusoidal, Synthetic, Absolute

# regarding Synthetic function: how to take into account randomness? See below for tips
# https://stackoverflow.com/questions/62757185/how-to-use-pytest-to-test-functions-that-generate-random-samples

cases_min = [
    (Synthetic, torch.tensor(0.5)),
    (Sinusoidal, torch.tensor(7.9787)),
    (Ackley, torch.zeros(2))
]

@pytest.mark.parametrize('case', cases_min, ids=lambda x: f'{x[0].__name__} {x[1:]}')
def test_minimum(case):
    loss_fn, minimum = case
    loss = loss_fn()
    assert torch.allclose(loss.minimum(), minimum)


cases_min_value = [
    (Synthetic, -.5),
    (Sinusoidal, -7.916727),
    (Ackley, 0.)
]

@pytest.mark.parametrize('case', cases_min_value, ids=lambda x: f'{x[0].__name__} {x[1:]}')
def test_min_value(case):
    loss_fn, value = case
    loss = loss_fn()
    min_loss = loss(loss.minimum())
    assert abs(min_loss.item() - value) <= 1e-6
