# -*- coding: utf-8 -*-
# @File    : ActivationForward.py
# @Author  : Hua Guo
# @Time    : 2020/2/5 上午8:24
# @Disc    :
import numpy as np


class ActivationForward:
    @staticmethod
    def sigmoid(z):
        """
        :param W:
        :param X:
        :param b:
        :return:
        """
        a = 1./(1+np.exp(-z))
        cache = z
        return a, cache

    @staticmethod
    def relu(z):
        cache = z
        return np.maximum(0, z), cache