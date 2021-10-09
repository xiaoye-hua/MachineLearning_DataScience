# -*- coding: utf-8 -*-
# @File    : test_FM.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 上午9:50
# @Disc    :
import tensorflow as tf
from unittest import TestCase

from src.Model.Recommender.FM import LinerPart, NonLinearPart, FM

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TestFM(TestCase):
    def setUp(self) -> None:
        self.batch_size = 1500
        self.feature_num = 34
        self.embeding_hidden_dim = 20
        self.x = tf.random.uniform(shape=(self.batch_size, self.feature_num))

    def test_LinearPart(self):
        linear_part = LinerPart()
        output = linear_part(self.x)
        self.assertTrue(output.shape == (self.batch_size, 1), output.shape)

    def test_NonLinearPart(self):
        nonlinear_part = NonLinearPart(feature_num=self.feature_num, hidden_dim=self.embeding_hidden_dim)
        output = nonlinear_part(self.x)
        self.assertTrue(output.shape == (self.batch_size, 1), output.shape)

    def test_FM(self):
        fm = FM(feature_num=self.feature_num, embed_dim=self.embeding_hidden_dim)
        output = fm(self.x)
        self.assertTrue(output.shape == (self.batch_size, 1), output.shape)

