import numpy as np
import torch
import math
from matplotlib import pyplot as plt
from optimal_pytorch import *
import time
"""
1. Class names are defined as : 'loss_' + func_name for pattern matching when generating list of all functions.
2. Each loss function should have 2 methods get_minima and forward which returns the minimum value of that function and returns a loss value
based on the input they receive.
3. Pass the config to calculate any coefficients while initializing the class in the __init__ method.
"""


# xs = starting point, xe = ending point, fprime = value of gradient at starting point, fs = value of function at ending point
# Quadratic function
class loss_quadratic:
    def __init__(self, opt):
        if (opt['xs'] >= opt['xe']):
            raise Exception('xs value greater than xe')
        if (opt['fprime'] < 0):
            raise Exception('fprime should be greater than 0')
        if (opt['fs'] < 0):
            raise Exception('fs should always be greater than 0')
        self.xs = opt['xs']
        self.xe = opt['xe']
        self.fprime = opt['fprime']
        self.fs = opt['fs']
        self.b = (self.xs + self.xe) / 2
        self.a = -self.fprime / (2 * (self.xs - self.b))
        self.c = self.fs - self.a * ((self.xs - self.b)**2)

    def get_minima(self):
        return self.b

    def forward(self, x):
        loss = self.a * (x - self.b)**2 + self.c
        return loss


# Absolute Value Function
class loss_absolute:
    def __init__(self, opt):
        if (opt['xs'] >= opt['xe']):
            raise Exception('xs value greater than xe')
        if (opt['fprime'] < 0):
            raise Exception('fprime should be greater than 0')
        if (opt['fs'] < 0):
            raise Exception('fs should always be greater than 0')
        self.xs = opt['xs']
        self.xe = opt['xe']
        self.fprime = opt['fprime']
        self.fs = opt['fs']
        self.b = (self.xs + self.xe) / 2
        self.a = self.fprime
        self.c = self.fs + self.a * (self.xs - self.b)

    def get_minima(self):
        return self.b

    def forward(self, x):
        if (x < self.b):
            loss = -self.a * (x - self.b) + self.c
        else:
            loss = self.a * (x - self.b) + self.c
        return loss


# mu = mean for the gaussian function pdf, std = standard deviation for gaussian function pdf
# Inverted gaussian pdf
class loss_gaussian:
    def __init__(self, opt):
        self.mu = opt['mu'] if ('mu' in opt) else torch.tensor(
            0, dtype=torch.float)  #some default value
        self.sd = opt['sd'] if ('sd' in opt) else torch.tensor(
            1, dtype=torch.float)
        self.coeff = 1 / (self.sd * (2 * math.pi)**0.5)

    def get_minima(self):
        return self.mu

    def forward(self, x):
        ex_coeff = -0.5 * (((x - self.mu) / self.sd)**2)
        loss = -self.coeff * torch.exp(ex_coeff)
        return loss


# -xsin(x) for x in [0, 10] defined in http://infinity77.net/global_optimization/test_functions_1d.html
class loss_x_sinx:
    def __init__(self, opt={}):
        self.opt = opt

    def get_minima(self):
        return torch.tensor(7.9787, dtype=torch.float)

    def forward(self, x):
        loss = -x * torch.sin(x)
        return loss


# synthetic function defined in https://arxiv.org/pdf/1912.01823.pdf
class loss_synthetic_func:
    def __init__(self, opt={}):
        self.opt = opt

    def get_minima(self):
        return torch.tensor(0.5, dtype=torch.float)

    def forward(self, x):
        temp = torch.rand(1, dtype=torch.float)
        if (temp < 0.002):
            loss = 999 * (x**2) / 2
        else:
            loss = -x
        return loss
