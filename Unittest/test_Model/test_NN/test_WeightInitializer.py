# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2020/2/3 下午11:11
# @Disc    :
from unittest import TestCase

from src.Model.NN.ParamsInitializer import ParamsInitializer


class TestWeightInitializer(TestCase):
    def test_random_initialization(self):
        dims = [2, 3]
        weights = ParamsInitializer.random_initialization(x_dim=dims[1], y_dim=dims[0])
        print(weights)
        self.assertTrue(
            weights.shape[0]== dims[1]
        )

    def test_zero_initialization(self):
        bias = ParamsInitializer.zero_initialization(3, 1)
        print(bias)