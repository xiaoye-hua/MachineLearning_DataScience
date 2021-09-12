# -*- coding: utf-8 -*-
# @File    : test_Kernel.py
# @Author  : Hua Guo
# @Time    : 2021/9/13 上午3:39
# @Disc    :
# -*- coding: utf-8 -*-
# @File    : test_kernel.py
# @Author  : Hua Guo
# @Time    : 2021/9/13 上午3:36
# @Disc    :
# -*- coding: utf-8 -*-
# @File    : test_kernel.py
# @Author  : Hua Guo
# @Time    : 2021/9/13 上午3:34
# @Disc    :
import numpy as np
from unittest import TestCase

from src.Model.SVM.kernel import kernel_rbf


class TestKernel(TestCase):
    def test_kernel_rbf(self):
        x = np.random.rand(100, 30)
        y = np.random.rand(200, 30)
        res = kernel_rbf(X=x, Y=y, k=1)
        print(res.shape)