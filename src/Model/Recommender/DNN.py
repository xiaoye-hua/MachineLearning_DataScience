# -*- coding: utf-8 -*-
# @File    : DNN.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午8:55
# @Disc    :
from src.BaseClass.DLModel import DLModel

from tensorflow.keras.layers import Dense, Activation
from typing import List


class DNN(DLModel):
    def __init__(self, hidden_dim: List[int], sigmoid=True) -> None:
        super(DNN, self).__init__()
        for dim in hidden_dim:
            self.layers_lst.append(
                Dense(units=dim, use_bias=True)
            )
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.layers_lst.append(
                Activation(activation="sigmoid")
            )