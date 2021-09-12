# -*- coding: utf-8 -*-
# @File    : KernelSVM.py
# @Author  : Hua Guo
# @Time    : 2021/9/12 下午11:51
# @Disc    : Thanks to https://github.com/cperales/SupportVectorMachine
import numpy as np
from functools import partial

from src.Model.SVM import BaseSVM
from src.Model.SVM.solver import fit_kernel
from src.Model.SVM.kernel import kernel_dict


class KernelSVM(BaseSVM):
    def __init__(self):
        """
        Refer to 统计学习方法-李航 算法7.4的公式
        """
        self.X = None
        self.y = None
        self.alphas = None
        self.b = None
        self.kernel = None

    def fit(self, X, y, kernel_type='rbf', k=1.0):
        y = self.change_label(y, replace={0.:-1.})
        # change datatype from int to float
        X = X.astype('float')
        y = y.astype('float')
        # partial function allows us to fix some arguments and get a new functions
        self.kernel = partial(kernel_dict[kernel_type.lower()], k=k)
        # Get alphas
        C = 1.0  # Penalty
        alphas = fit_kernel(X, y, C, self.kernel)
        # normalize
        alphas = alphas / np.linalg.norm(alphas)
        # Get b
        b_vector = y - np.sum(self.kernel(X) * alphas * y[:, None], axis=0)
        b = b_vector.sum() / b_vector.size
        # Store values
        self.X = X
        self.y = y
        self.alphas = alphas
        self.b = b

    def predict(self, X):
        prod = np.sum(self.kernel(self.X, X) * self.alphas * self.y[:, None],
                      axis=0) + self.b
        res = np.sign(prod)
        return self.change_label(res, replace={-1.:0.})