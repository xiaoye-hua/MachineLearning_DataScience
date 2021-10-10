# -*- coding: utf-8 -*-
# @File    : DeepFM.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午8:47
# @Disc    :
from src.BaseClass.DLModel import DLModel
from src.Model.Recommender.FM import FM
from src.Model.Recommender.DNN import DNN

from tensorflow.keras.layers import Activation
from typing import List
import tensorflow as tf


class DeepFM(DLModel):
    def __init__(self, feature_num: int, embed_dim: int, dnn_dims: List[int], output_dim=1) -> None:
        super(DeepFM, self).__init__()
        self.fm = FM(feature_num=feature_num, embed_dim=embed_dim, output_dim=output_dim, sigmoid=False)
        self.dnn = DNN(hidden_dim=dnn_dims, sigmoid=False)
        self.ac = Activation(activation='sigmoid')

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        output = self.ac(self.fm(inputs) + self.dnn(inputs))
        return output