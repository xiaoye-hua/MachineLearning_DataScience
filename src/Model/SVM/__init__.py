# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午9:54
# @Disc    :
import numpy as np


class BaseSVM:
    def change_label(self, data, replace):
        # change label from -1, 1 to 0, 1
        # replace = {-1: 1}
        replace = np.array([list(replace.keys()), list(replace.values())])    # Create 2D replacement matrix
        mask = np.in1d(data, replace[0, :])                                   # Find elements that need replacement
        data[mask] = replace[1, np.searchsorted(replace[0, :], data[mask])]
        return data

