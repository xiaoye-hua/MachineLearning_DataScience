# -*- coding: utf-8 -*-
# @File    : WeightInitializer.py
# @Author  : Hua Guo
# @Time    : 2020/2/3 下午11:08
# @Disc    :
import numpy as np
from typing import List, Type


class ParamsInitializer:
    """
    There are two types of parameters to initialize in a neural network:
        - the weight matrices $(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
        - the bias vectors $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$

    A well chosen initialization can:
        - Speed up the convergence of gradient descent
        - Increase the odds of gradient descent converging to a lower training (and generalization) error
    """
    @staticmethod
    def random_initialization(x_dim: int, y_dim: int) -> Type[np.array]:
        """
        random select from normal distribution
        :param dims:
        :return:
        """
        return np.random.randn(x_dim, y_dim) * 0.001

    @staticmethod
    def zero_initialization(x_dim: int, y_dim: int) -> Type[np.array]:
        """
        :param output_dim:
        :param input_dim:
        :return:
        """
        return np.zeros([x_dim, y_dim])

    @staticmethod
    def he_initialization(x_dim: int, y_dim: int)-> Type[np.array]:
        """
        this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization",
         this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])`
          where He initialization would use `sqrt(2./layers_dims[l-1])`.)
        :param x_dim:
        :param y_dim:
        :return:
        """
        return np.random.randn(x_dim, y_dim) * np.sqrt(2./y_dim)

    @staticmethod
    def Xavier_initialization(x_dim: int, y_dim: int, random_seed=None)-> Type[np.array]:
        if random_seed is not None:
            np.random.seed(random_seed)
        return np.random.randn(x_dim, y_dim) / np.sqrt(y_dim)

