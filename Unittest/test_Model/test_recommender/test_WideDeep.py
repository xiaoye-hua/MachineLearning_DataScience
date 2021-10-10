# -*- coding: utf-8 -*-
# @File    : test_WideDeep.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午9:34
# @Disc    :
import tensorflow as tf
from unittest import TestCase
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.Model.Recommender.WideDeep import WideDeep


class TestDeepFM(TestCase):
    def setUp(self) -> None:
        self.batch_size = 1500
        self.feature_num = 34
        self.embeding_hidden_dim = 20
        self.x = tf.random.uniform(shape=(self.batch_size, self.feature_num))

    def test_DeepFM(self):
        model = WideDeep(dnn_dims=[124, 64, 1], output_dim=1)
        output = model(self.x)
        self.assertTrue(output.shape == (self.batch_size, 1), output.shape)