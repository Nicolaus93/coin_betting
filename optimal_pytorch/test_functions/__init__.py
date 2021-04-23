"""
`optimal_pytorch.test_functions` is a package implementing
various test functions for optimization algorithms.
"""
from .loss import Ackley, Absolute, Quadratic, Sinusoidal, Synthetic, InvGaussian

__all__ = ["Ackley", "Absolute", "Quadratic", "Sinusoidal", "Synthetic", "InvGaussian"]
