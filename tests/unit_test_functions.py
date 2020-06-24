from functions import *
import numpy as np
import torch
import unittest
from math import pi


class Test_Functions(unittest.TestCase):
    def test_quadratic(self):
        # This config is for y = 3(x - 2)**2 + 1
        self.assertEqual(
            loss_quadratic({
                'xs': 0,
                'xe': 4,
                'fprime': 12,
                'fs': 13
            }).forward(torch.tensor(1)), torch.tensor(4))
        self.assertEqual(
            loss_quadratic({
                'xs': 0,
                'xe': 4,
                'fprime': 12,
                'fs': 13
            }).get_minima(), 2)
        #Test where xs > xe
        with self.assertRaises(Exception):
            loss_quadratic({
                'xs': 4,
                'xe': 0,
                'fprime': 12,
                'fs': 13
            }).forward(torch.tensor(1)), torch.tensor(4)
        #Test where fprime is negative
        with self.assertRaises(Exception):
            loss_quadratic({
                'xs': 0,
                'xe': 4,
                'fprime': -12,
                'fs': 13
            }).forward(torch.tensor(1)), torch.tensor(4)
        #Test where fs is negative
        with self.assertRaises(Exception):
            loss_quadratic({
                'xs': 0,
                'xe': 4,
                'fprime': 12,
                'fs': -13
            }).forward(torch.tensor(1)), torch.tensor(4)
        # This config is for y = 4(x - 3)**2 + 2
        self.assertEqual(
            loss_quadratic({
                'xs': -1,
                'xe': 7,
                'fprime': 32,
                'fs': 66
            }).forward(torch.tensor(0)), torch.tensor(38))
        self.assertEqual(
            loss_quadratic({
                'xs': -1,
                'xe': 7,
                'fprime': 32,
                'fs': 66
            }).get_minima(), 3)

    def test_absolute(self):
        # This config is for y = 3|x - 2| + 1
        self.assertEqual(
            loss_absolute({
                'xs': 0,
                'xe': 4,
                'fprime': 3,
                'fs': 7
            }).forward(torch.tensor(1)), torch.tensor(4))
        self.assertEqual(
            loss_absolute({
                'xs': 0,
                'xe': 4,
                'fprime': 3,
                'fs': 7
            }).get_minima(), 2)
        #Test where xs > xe
        with self.assertRaises(Exception):
            loss_absolute({
                'xs': 4,
                'xe': 0,
                'fprime': 3,
                'fs': 7
            }).forward(torch.tensor(1)), torch.tensor(4)
        #Test where fprime is negative
        with self.assertRaises(Exception):
            loss_absolute({
                'xs': 0,
                'xe': 4,
                'fprime': -3,
                'fs': 7
            }).forward(torch.tensor(1)), torch.tensor(4)
        #Test where fs is negative
        with self.assertRaises(Exception):
            loss_absolute({
                'xs': 0,
                'xe': 4,
                'fprime': 3,
                'fs': -7
            }).forward(torch.tensor(1)), torch.tensor(4)
        # This config is for y = 4|x - 3| + 2
        self.assertEqual(
            loss_absolute({
                'xs': -1,
                'xe': 7,
                'fprime': 4,
                'fs': 18
            }).forward(torch.tensor(0)), torch.tensor(14))
        self.assertEqual(
            loss_absolute({
                'xs': -1,
                'xe': 7,
                'fprime': 4,
                'fs': 18
            }).get_minima(), 3)

    def test_gaussian(self):
        # Default case mu = 0, sd = 1, for x = 0 will give 1/(2*pi)**0.5
        self.assertEqual(
            loss_gaussian({}).forward(torch.tensor(0)),
            torch.tensor(-1 / (2 * math.pi)**0.5, dtype=torch.float))


unittest.main()
