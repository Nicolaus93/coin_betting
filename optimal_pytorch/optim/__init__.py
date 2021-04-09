"""
Created on Tue Feb 11th, 2020

@author: Zhenxun Zhuang
"""
"""
:mod:`optimal_pytorch` is a package implementing various optimization algorithms.
The interface is consistent with the torch.optim package.
"""

from .adam import Adam                           # noqa: F401
from .sgd import SGD                             # noqa: F401
from .sgdol import SGDOL                         # noqa: F401
from .optimizer import Optimizer                 # noqa: F401
from .accsgd import AccSGD                       # noqa: F401
from .adabound import AdaBound                   # noqa: F401
from .adamod import AdaMod                       # noqa: F401
from .diffgrad import DiffGrad                   # noqa: F401
from .lamb import Lamb                           # noqa: F401
from .lookahead import Lookahead                 # noqa: F401
from .novograd import NovoGrad                   # noqa: F401
from .pid import PID                             # noqa: F401
from .radam import RAdam                         # noqa: F401
from .sgdw import SGDW                           # noqa: F401
from .yogi import Yogi                           # noqa: F401
from .cocob import Cocob                         # noqa: F401
from .coin_betting import CoinBetting            # noqa: F401


# del adam
# del sgd
# del sgdol
# del optimizer

__all__ = [
    'Adam',
    'SGD',
    'SGDOL',
    'Optimizer',
    'AccSGD',
    'AdaBound',
    'AdaMod',
    'Cocob',
    'DiffGrad',
    'Lamb',
    'Lookahead',
    'NovoGrad',
    'PID',
    'RAdam',
    'SGDW',
    'Yogi',
    'CoinBetting']
