# -*- coding: utf-8 -*-
# @File    : LinearSVM.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午10:34
# @Disc    : Inspired from https://github.com/cperales/SupportVectorMachine
import numpy as np

from src.Model.SVM.solver import fit_soft, fit
from src.Model.SVM import BaseSVM


class LinearSVM(BaseSVM):
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, soft=False):
        y = self.change_label(y, replace={0.:-1.})
        # change datatype from int to float
        X = X.astype('float')
        y = y.astype('float')
        if soft:
            # penalty
            C = 1
            alphas = fit_soft(x=X, y=y, C=C)
        else:
            alphas = fit(x=X, y=y)
        # Refer to 统计学习方法-李航 算法7.2
        self.w = np.sum(alphas*y[:, None]*X, axis=0)
        self.b = y - np.dot(self.w, X.T)
        self.b = sum(self.b)/self.b.size
        # nomalization
        norm = np.linalg.norm(self.w)
        self.w = self.w/norm
        self.b = self.b/norm

    def predict(self, X):
        y = np.sign(np.dot(self.w, X.T) + self.b*np.ones(X.shape[0]))
        return self.change_label(y, replace={-1.:0.})

