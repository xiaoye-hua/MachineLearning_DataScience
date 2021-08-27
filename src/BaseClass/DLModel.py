# -*- coding: utf-8 -*-
# @File    : DLModel.py
# @Author  : Hua Guo
# @Time    : 2021/8/14 上午8:12
# @Disc    :
from abc import abstractmethod
from tensorflow.keras import Model


class DLModel(Model):
    def __init__(self):
        super(DLModel, self).__init__()
        self.layers_lst = []

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        for layer in self.layers_lst:
            inputs = layer(inputs)
        return inputs

