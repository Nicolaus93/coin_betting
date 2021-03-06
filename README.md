# optimal-pytorch

![Build Status](https://github.com/Nicolaus93/coin_betting/actions/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/Nicolaus93/coin_betting/branch/master/graph/badge.svg)](https://codecov.io/gh/Nicolaus93/coin_betting)
[![Maintainability](https://api.codeclimate.com/v1/badges/62dcc62f012165d75a7f/maintainability)](https://codeclimate.com/github/Nicolaus93/coin_betting/maintainability)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nicolaus93/coin_betting/animation.py)



<!-- Badges -->
[build-image]: https://github.com/Nicolaus93/coin_betting/workflows/build.yml/badge.svg
[build-url]: https://github.com/Nicolaus93/coin_betting/actions/workflows/build.yml


A library which combines Coin-Betting algorithms and test functions for optimization algorithms.

Install it with `pip install optimal_pytorch`.

## Usage

A minimal example is shown below

```
import torch
from optimal_pytorch.coin_betting.torch import Cocob

loss_f = lambda x: x**2
y = torch.tensor([20.0], requires_grad=True)
optimizer = Cocob([y])
iterations = 50
for step in range(iterations):
    optimizer.zero_grad()
    loss = loss_f(y)
    loss.backward()
    optimizer.step()
    print(y)
```
