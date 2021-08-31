# -*- coding: utf-8 -*-
# @File    : test_Seq2Seq.py
# @Author  : Hua Guo
# @Time    : 2021/8/30 下午12:37
# @Disc    :
import tensorflow as tf
from unittest import TestCase

from src.Model.SequenceModel.Seq2Seq import Seq2SeqEncoder


class TestSeq2Seq(TestCase):
    def test_Seq2SeqEncoder(self):
        encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                 num_layers=2)
        X = tf.zeros((4, 7))
        output, state = encoder(X, training=False)
        self.assertTrue(output.shape == [4, 7, 16])
        self.assertTrue(state[0].shape == [4, 16])
        self.assertTrue(state[1].shape == [4, 16])
