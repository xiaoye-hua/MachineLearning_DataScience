# -*- coding: utf-8 -*-
# @File    : MF.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午9:32
# @Disc    : Matrix Factorization
from tensorflow.keras.layers import Embedding
import tensorflow as tf
from typing import Dict

from tensorflow.keras import Model


class MF(Model):
    def __init__(self, user_num: int, item_num: int, hidden_dim: int) -> None:
        super(MF, self).__init__()
        self.user_vector = Embedding(input_dim=user_num, output_dim=hidden_dim)
        self.item_vector = Embedding(input_dim=item_num, output_dim=hidden_dim)
        self.user_bias = Embedding(input_dim=user_num, output_dim=1)
        self.item_bias = Embedding(input_dim=item_num, output_dim=1)

    def call(self, inputs: Dict[str, int], training=None, mask=None):
        user_id = inputs["user_id"]
        item_id = inputs['item_id']
        user_rep = self.user_vector(user_id)
        item_rep = self.item_vector(item_id)
        user_bias = self.user_bias(user_id)
        item_bias = self.item_bias(item_id)
        rate = tf.matmul(tf.expand_dims(user_rep, 0), tf.expand_dims(item_rep, -1)) + user_bias + item_bias
        return rate