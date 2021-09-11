# -*- coding: utf-8 -*-
# @File    : ActivationBackward.py
# @Author  : Hua Guo
# @Time    : 2020/2/6 上午8:35
# @Disc    :
import numpy as np


class ActivationBackward:
    @staticmethod
    def sigmoid_backward(dA, cache):
        """
        :param dA: derivative of cost with respect to A
        :param cache:
        :return: dZ: derivative of cost with respect to Z
        """
        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def relu_backward(dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ