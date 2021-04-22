import torch
from abc import ABC, abstractmethod
from math import pi, exp


class GenericLoss(ABC):
    """
    Each loss function should have 2 methods:
     - minimum:  returns the minimum value of that function
     - __call__:     returns a loss value based on the input it receives.
    """

    @abstractmethod
    def minimum(self) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Ackley(GenericLoss):
    """
    The Ackley function is widely used for testing optimization algorithms.
    It is characterized by a nearly flat outer region, and a large hole at the centre.
    See https://www.sfu.ca/~ssurjano/ackley.html

    References:
        - Adorio, E. P., & Diliman, U. P. MVF
        Multivariate Test Functions Library in C for Unconstrained Global Optimization (2005).
        Retrieved June 2013, from http://http://www.geocities.ws/eadorio/mvf.pdf.

        - Molga, M., & Smutnicki, C.
        Test functions for optimization needs (2005).
        Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.

        - Back, T. (1996).
        Evolutionary algorithms in theory and practice: evolution strategies,
        evolutionary programming, genetic algorithms. Oxford University Press on Demand.
    """

    def __init__(self, a: float = 20., b: float = .2, c: float = 2 * pi):
        self.a = a
        self.b = b
        self.c = c
        self.dim = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[0]
        if d != self.dim:
            self.dim = d
        s1 = torch.norm(x / d)
        s2 = torch.cos(self.c * x).sum() / d
        return -self.a * torch.exp(-self.b * s1) - torch.exp(s2) + self.a + exp(1)

    def minimum(self) -> torch.Tensor:
        if self.dim:
            return torch.zeros(self.dim)
        return torch.zeros(2)


class Absolute(GenericLoss):
    """
    Absolute loss Function in 1d.
    f(x) = a * |x - b| + c.
    """

    def __init__(self, a: float = 1., b: float = 1., c: float = 1.):
        self.a = a
        self.b = b
        self.c = c

    def minimum(self) -> torch.Tensor:
        return torch.tensor(self.b, dtype=torch.float)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.abs(x - self.b) + self.c

    def __repr__(self) -> str:
        a, b, c = self.a, self.b, self.c
        sign = {True: '+', False: '-'}
        return f"f(x) = {a:.2f} * |x {sign[b < 0]} {abs(b)}| {sign[c > 0]} {abs(c)}"


class Quadratic(GenericLoss):
    """
    Quadratic loss function in 1d.
    f(x) = a * (x - b)^2 + c.
    """

    def __init__(self, a: float = 1., b: float = 1., c: float = 1.):
        self.a = a
        self.b = b
        self.c = c

    def minimum(self) -> torch.float:
        return self.b

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x - self.b)**2 + self.c

    def __repr__(self) -> str:
        sign = {True: '+', False: '-'}
        return f"f(x) = {a:.2f} * (x {sign[b > 0]} {abs(b)})^2 {sign[c > 0]} {abs(c)}"


class Sinusoidal(GenericLoss):
    """
    Function defined in http://infinity77.net/global_optimization/test_functions_1d.html
    See Problem 10.
        f(x) = -x * sin(x), for x in [0, 10].
    """

    def __init__(self):
        return

    def minimum(self) -> torch.float:
        return torch.tensor(7.9787, dtype=torch.float)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return -x * torch.sin(x)

    def __repr__(self) -> str:
        return "f(x) = -x * sin(x)"


class Synthetic(GenericLoss):
    """
    Synthetic function defined in https://arxiv.org/pdf/1912.01823.pdf
    """

    def __init__(self):
        return

    def minimum(self) -> torch.float:
        # this is not a minimum but rather a stationary point
        return torch.tensor(0.5, dtype=torch.float)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sample = torch.rand(1, dtype=torch.float)
        if sample < 0.002:
            loss = 999 * (x**2) / 2
        else:
            loss = -x
        return loss

    def __repr__(self) -> str:
        return f"f(x) = 999 * x^2 / 2 with prob 0.002, -x otherwise."


class InvGaussian(GenericLoss):
    """
    Inverted Gaussian pdf.
    mu = mean for the gaussian function pdf
    std = standard deviation for gaussian function pdf
    """

    def __init__(self, mu: float = 0., std: float = 1.):
        self.mu = mu
        self.std = std

    def minimum(self) -> float:
        return self.mu

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.std
        mu = self.mu
        return -torch.exp((-.5 * (x - mu) / sigma)**2) / (sigma * (2 * pi)**.5)

    def __repr__(self) -> str:
        return f"Inverted Gaussian pdf centered at {self.mu} with sigma={self.std}"
