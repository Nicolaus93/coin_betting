import torch
from abc import ABC, abstractmethod
from typing import Mapping
from math import pi
from typing import NewType

torch_float = NewType('torch_float', torch.float)


class generic_loss(ABC):
    """
    Each loss function should have 2 methods:
     - get_minima, which returns the minimum value of that function
     - forward, which returns a loss value based on the input they receive.
    """

    @abstractmethod
    def get_minima(self) -> torch_float:
        pass

    @abstractmethod
    def forward(self, x: torch.tensor) -> torch.tensor:
        pass


class absolute_loss(generic_loss):
    """
    Absolute loss Function in 1d.
    f(x) = a * |x - b| + c.
    Given a and interval [x1, x2]:
        b = (x1 + x2) / 2
        c = offset - a * | x2 - b |
    so that f(x2) = offset.
    """

    def __init__(self, opt: Mapping[str, torch_float]) -> None:
        if (opt['xs'] >= opt['xe']):
            raise ValueError('xs value greater than xe')
        if (opt['slope'] < 0):
            raise ValueError('slope should be greater than 0')
        if (opt['offset'] < 0):
            raise ValueError('offset should always be greater than 0')
        self.xs = opt['xs']
        self.xe = opt['xe']
        self.b = (self.xs + self.xe) / 2
        self.a = opt['slope']
        self.c = opt['offset'] - self.a * abs(self.xe - self.b)
        self.name = "abs_loss"

    def get_minima(self) -> torch_float:
        return self.b

    def forward(self, x: torch.tensor) -> torch.tensor:
        if (x < self.b):
            loss = -self.a * (x - self.b) + self.c
        else:
            loss = self.a * (x - self.b) + self.c
        return loss

    def __repr__(self) -> str:
        return "Absolute loss function in 1d."


class quadratic_loss(generic_loss):
    """
    Quadratic loss function in 1d.
    f(x) = a * (x - b)^2 + c.
    Given a and interval [x1, x2]:
        b = (x1 + x2) / 2
        c = offset - a * (x2 - b)^2
    so that f(x2) = offset.
    """

    def __init__(self, opt: Mapping[str, torch_float]) -> None:
        if opt['xs'] >= opt['xe']:
            raise ValueError('xs value greater than xe')
        if opt['slope'] < 0:
            raise ValueError('slope should be greater than 0')
        if opt['offset'] < 0:
            raise ValueError('offset should always be greater than 0')
        self.xs = opt['xs']
        self.xe = opt['xe']
        self.slope = opt['slope']
        self.b = (self.xs + self.xe) / 2
        self.a = -self.slope / (2 * (self.xs - self.b))
        self.c = opt['offset'] - self.a * ((self.xs - self.b)**2)
        self.name = "quadratic_loss"

    def get_minima(self) -> torch_float:
        return self.b

    def forward(self, x: torch.tensor) -> torch.tensor:
        loss = self.a * (x - self.b)**2 + self.c
        return loss

    def __repr__(self) -> str:
        res = "Quadratic loss function in 1d:\n\tf(x) = {:.2f} * (x - {})^2".format(
            self.a, self.b)
        # following: same can happen for b?
        if self.c > 0:
            return res + " + {}".format(self.c)
        elif self.c < 0:
            return res + " - {}".format(abs(self.c))
        else:
            return res


class sinusoidal_loss(generic_loss):
    """
    Function defined in http://infinity77.net/global_optimization/test_functions_1d.html
    f(x) = -x * sin(x), for x in [0, 10].
    """

    def __init__(self, opt):
        self.opt = opt
        self.name = "sinusoidal_loss"

    def get_minima(self) -> torch_float:
        return torch.tensor(7.9787, dtype=torch_float)

    def forward(self, x: torch.tensor) -> torch.tensor:
        loss = -x * torch.sin(x)
        return loss


class synthetic_loss(generic_loss):
    """
    Synthetic function defined in https://arxiv.org/pdf/1912.01823.pdf
    """

    def __init__(self, opt):
        self.opt = opt
        self.name = "synthetic_loss"

    def get_minima(self) -> torch_float:
        return torch.tensor(0.5, dtype=torch_float)

    def forward(self, x: torch.tensor) -> torch.tensor:
        temp = torch.rand(1, dtype=torch_float)
        if (temp < 0.002):
            loss = 999 * (x**2) / 2
        else:
            loss = -x
        return loss


class gaussian_loss(generic_loss):
    """
    Inverted gaussian pdf
    mu = mean for the gaussian function pdf
    std = standard deviation for gaussian function pdf
    """

    def __init__(self, opt: Mapping[str, torch_float]) -> None:
        self.mu = opt['mu'] if ('mu' in opt) else torch.tensor(
            0, dtype=torch_float)  # some default value
        self.sd = opt['sd'] if ('sd' in opt) else torch.tensor(
            1, dtype=torch_float)
        self.coeff = 1 / (self.sd * (2 * pi)**0.5)
        self.name = "gaussian_loss"

    def get_minima(self) -> float:
        return self.mu

    def forward(self, x):
        exp_coeff = -0.5 * (((x - self.mu) / self.sd)**2)
        loss = -self.coeff * torch.exp(exp_coeff)
        return loss
