"""
Created on Tue Feb 11th, 2020

@author: Zhenxun Zhuang
"""
"""
:mod:`optimal_pytorch` is a package implementing various optimization algorithms.
The interface is consistent with the torch.optim package.
"""

from .adam import Adam  # noqa: F401
from .sgd import SGD  # noqa: F401
from .sgdol import SGDOL  # noqa: F401
from .optimizer import Optimizer  # noqa: F401
from .accsgd import AccSGD
from .adabound import AdaBound
from .adamod import AdaMod
from .diffgrad import DiffGrad
from .lamb import Lamb
from .lookahead import Lookahead
from .novograd import NovoGrad
from .pid import PID
from .radam import RAdam
from .sgdw import SGDW
from .yogi import Yogi


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
    'DiffGrad',
    'Lamb',
    'Lookahead',
    'NovoGrad',
    'PID',
    'RAdam',
    'SGDW',
    'Yogi']
