# -*- coding: utf-8 -*-
# @File    : WideDeep.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午9:27
# @Disc    :
from src.BaseClass.DLModel import DLModel
from src.Model.Recommender.DNN import DNN, Dense

from tensorflow.keras.layers import Activation
from typing import List
import tensorflow as tf


class WideDeep(DLModel):
    def __init__(self, dnn_dims: List[int], output_dim=1) -> None:
        super(WideDeep, self).__init__()
        self.wide = Dense(units=output_dim, use_bias=True)
        self.deep = DNN(hidden_dim=dnn_dims, sigmoid=False)
        self.ac = Activation(activation='sigmoid')

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        final = self.ac(self.deep(inputs) + self.wide(inputs))
        return final