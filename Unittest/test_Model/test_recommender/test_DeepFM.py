# -*- coding: utf-8 -*-
# @File    : test_DeepFM.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午8:46
# @Disc    :
import tensorflow as tf
from unittest import TestCase
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from src.Model.Recommender.DeepFM import DeepFM


class TestDeepFM(TestCase):
    def setUp(self) -> None:
        self.batch_size = 1500
        self.feature_num = 34
        self.embeding_hidden_dim = 20
        self.x = tf.random.uniform(shape=(self.batch_size, self.feature_num))

    def test_DeepFM(self):
        model = DeepFM(feature_num=self.feature_num, embed_dim=self.embeding_hidden_dim,
                       dnn_dims=[64, 128, 64, 1])
        output = model(self.x)
        self.assertTrue(output.shape == (self.batch_size, 1), output.shape)