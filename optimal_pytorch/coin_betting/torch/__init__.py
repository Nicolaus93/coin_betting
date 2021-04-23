"""
`optimal_pytorch.coin_betting` is a package implementing
various coin-betting algorithms. The interface is consistent
with the torch.optim package.
"""
from .sgdol import SGDOL  # noqa: F401
from .cocob import Cocob  # noqa: F401
from .recursive import Recursive  # noqa: F401
from .regralizer import Regralizer  # noqa: F401
from .scinol2 import Scinol2  # noqa: F401
from .onsbet import ONSBet  # noqa: F401


__all__ = ["SGDOL", "Cocob", "Recursive", "Regralizer", "Scinol2", "ONSBet"]
