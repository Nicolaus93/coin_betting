"""
This module contains implementations of various test functions.
"""
from abc import ABC, abstractmethod
from math import pi, exp
import torch


class GenericLoss(ABC):
    """
    Each loss function should have 2 methods:
     - minimum:  returns the minimum value of that function
     - __call__:     returns a loss value based on the input it receives.
    """

    @abstractmethod
    def minimum(self) -> torch.Tensor:
        """
        Returns the point where the value of the loss function is minimized.
        """

    @abstractmethod
    def __call__(self, input_point: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the function in input_point.
        """


class Ackley(GenericLoss):
    """
    The Ackley function is widely used for testing optimization algorithms.
    It is characterized by a nearly flat outer region, and a large hole at the centre.
    See https://www.sfu.ca/~ssurjano/ackley.html

    References:
        - Adorio, E. P., & Diliman, U. P. MVF
        Multivariate Test Functions Library in C for Unconstrained
        Global Optimization (2005), from http://http://www.geocities.ws/eadorio/mvf.pdf.

        - Molga, M., & Smutnicki, C.
        Test functions for optimization needs (2005).
        From http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.

        - Back, T. (1996).
        Evolutionary algorithms in theory and practice: evolution strategies,
        evolutionary programming, genetic algorithms. Oxford University Press on Demand.
    """

    def __init__(self, a: float = 20.0, b: float = 0.2, c: float = 2 * pi):
        self.slope = a
        self.offset = b
        self.bias = c
        self.dim = 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[0]
        if dim != self.dim:
            self.dim = dim
        sum1 = torch.norm(x / dim)
        sum2 = torch.cos(self.bias * x).sum() / dim
        return (
            -self.slope * torch.exp(-self.offset * sum1)
            - torch.exp(sum2)
            + self.slope
            + exp(1)
        )

    def minimum(self) -> torch.Tensor:
        return torch.zeros(self.dim)


class Absolute(GenericLoss):
    """
    Absolute loss Function in 1d.
    f(x) = a * |x - b| + c.
    """

    def __init__(self, slope: float = 1.0, offset: float = 1.0, bias: float = 1.0):
        self.slope = slope
        self.offset = offset
        self.bias = bias

    def minimum(self) -> torch.Tensor:
        return torch.tensor(self.offset)

    def __call__(self, input_point: torch.Tensor) -> torch.Tensor:
        return self.slope * torch.abs(input_point - self.offset) + self.bias

    def __repr__(self) -> str:
        slope = self.slope
        offset = self.offset
        bias = self.bias
        sign = {True: "+", False: "-"}
        return "f(x) = {:.2f} * |x {} {}| {} {}".format(
            slope, sign[offset < 0], abs(offset), sign[bias > 0], abs(bias)
        )


class Quadratic(GenericLoss):
    """
    Quadratic loss function in 1d.
    f(x) = a * (x - b)^2 + c.
    """

    def __init__(self, slope: float = 1.0, offset: float = 1.0, bias: float = 1.0):
        self.slope = slope
        self.offset = offset
        self.bias = bias

    def minimum(self) -> torch.Tensor:
        return torch.tensor(self.offset)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.slope * (x - self.offset) ** 2 + self.bias

    def __repr__(self) -> str:
        slope = self.slope
        offset = self.offset
        bias = self.bias
        sign = {True: "+", False: "-"}
        return "f(x) = {:.2f} * (x {} {})^2 {} {}".format(
            slope, sign[offset > 0], abs(offset), sign[bias > 0], abs(bias)
        )


class Sinusoidal(GenericLoss):
    """
    Function defined in http://infinity77.net/global_optimization/test_functions_1d.html
    See Problem 10.
        f(x) = -x * sin(x), for x in [0, 10].
    """

    def __init__(self):
        return

    def minimum(self) -> torch.Tensor:
        return torch.tensor(7.9787)

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

    def minimum(self) -> torch.Tensor:
        # this is not a minimum but rather a stationary point
        return torch.tensor(0.5)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sample = torch.rand(1, dtype=torch.float)
        if sample < 0.002:
            loss = 999 * (x ** 2) / 2
        else:
            loss = -x
        return loss

    def __repr__(self) -> str:
        return "f(x) = 999 * x^2 / 2 with prob 0.002, -x otherwise."


class InvGaussian(GenericLoss):
    """
    Inverted Gaussian pdf.
    mu = mean for the gaussian function pdf
    std = standard deviation for gaussian function pdf
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def minimum(self) -> torch.Tensor:
        return torch.tensor(self.mean)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.std
        mean = self.mean
        return -torch.exp((-0.5 * (x - mean) / sigma) ** 2) / (sigma * (2 * pi) ** 0.5)

    def __repr__(self) -> str:
        return f"Inverted Gaussian pdf centered at {self.mean} with sigma={self.std}"
