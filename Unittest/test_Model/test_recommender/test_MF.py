# -*- coding: utf-8 -*-
# @File    : test_MF.py
# @Author  : Hua Guo
# @Time    : 2021/10/9 下午10:21
# @Disc    :
from unittest import TestCase

from src.Model.Recommender.MF import MF


class TestMF(TestCase):
    def test_MF(self):
        user_num = 10
        item_num = 20
        hidden_dim = 15
        inputs = {
            'user_id': 3
            , 'item_id': 4
        }
        model = MF(user_num=user_num, item_num=item_num, hidden_dim=hidden_dim)
        rate = model(inputs)
        self.assertTrue(rate.shape == (1, 1))
