"""
This is adapted from https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/types.py

Created on June 30th, 2020

@author: Nico
"""
from typing import Iterable, Union, Callable, Dict, Optional, Tuple, Any
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
LossClosure = Callable[[], float]
State = Dict[str, Any]
Betas2 = Tuple[float, float]
Nus2 = Tuple[float, float]
# OptLossClosure = Optional[LossClosure]
# OptFloat = Optional[float]