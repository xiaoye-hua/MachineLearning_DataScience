# -*- coding: utf-8 -*-
# @File    : LinearSVM.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午10:34
# @Disc    :
import numpy as np

from src.Model.SVM.solver import fit_soft, fit


class LinearSVM:
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

    def change_label(self, data, replace):
        # change label from -1, 1 to 0, 1
        # replace = {-1: 1}
        replace = np.array([list(replace.keys()), list(replace.values())])    # Create 2D replacement matrix
        mask = np.in1d(data, replace[0, :])                                   # Find elements that need replacement
        data[mask] = replace[1, np.searchsorted(replace[0, :], data[mask])]
        return data