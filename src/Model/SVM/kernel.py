# -*- coding: utf-8 -*-
# @File    : kernel.py
# @Author  : Hua Guo
# @Time    : 2021/9/13 上午3:28
# @Disc    :
import numpy as np


def kernel_rbf(X, Y=None, k=1):
    """

    :param X: array with n elements
    :param Y: (optional) array with m elements,
        same attributes
    :param k:
    :return:
    """
    n = X.shape[0]
    if Y is None:
        XX = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1),
                    np.ones((1, n)))
        XXh = XX + XX.T - 2 * np.dot(X, X.T)
    else:
        m = Y.shape[0]
        XX_1 = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1),
                      np.ones((1, m)))
        XX_2 = np.dot(np.sum(np.power(Y, 2), 1).reshape(m, 1),
                      np.ones((1, n)))
        XXh = XX_1 + XX_2.T - 2 * np.dot(X, Y.T)
    return np.exp(-XXh / k)


kernel_dict = {
    "rbf" : kernel_rbf
}