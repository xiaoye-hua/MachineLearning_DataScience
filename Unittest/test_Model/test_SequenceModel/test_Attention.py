# -*- coding: utf-8 -*-
# @File    : test_Attention.py
# @Author  : Hua Guo
# @Time    : 2021/8/30 上午10:52
# @Disc    :
import tensorflow as tf
from unittest import TestCase

from src.Model.SequenceModel.Attention import masked_softmax, AdditiveAttention, DotProductAttention, MultiHeadAttention


class TestSeq2SeqAttention(TestCase):

    def test_masked_softmax(self):
        X = tf.random.uniform(shape=(2, 2, 4))
        valid_len = tf.constant([2, 3])
        res = masked_softmax(X=X, valid_lens=valid_len)
        print(res)
        self.assertTrue(res.shape==X.shape)

    def test_AdditiveAttention(self):
        """
        query and key are vectors of different length
        Returns:
        """
        query_size = 20
        key_size = 20
        queries, keys = tf.random.normal(shape=(2, 1, query_size)), tf.ones((2, 10, key_size))
        # The two value matrices in the `values` minibatch are identical
        values = tf.repeat(
            tf.reshape(tf.range(40, dtype=tf.float32), shape=(1, 10, 4)), repeats=2,
            axis=0)
        valid_lens = tf.constant([2, 6])
        attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                      dropout=0.1)
        output = attention(queries, keys, values, valid_lens, training=False)
        # output.shape
        self.assertTrue(output.shape==[2, 1, 4])

    def test_DotProductAttention(self):
        """
        query and key are vectors of same length
        Returns:
        """
        query_size = key_size = 2
        queries, keys = tf.random.normal(shape=(2, 1, query_size)), tf.ones((2, 10, key_size))
        # The two value matrices in the `values` minibatch are identical
        values = tf.repeat(
            tf.reshape(tf.range(40, dtype=tf.float32), shape=(1, 10, 4)), repeats=2,
            axis=0)
        valid_lens = tf.constant([2, 6])
        attention = DotProductAttention(
            # num_hiddens=8,
                                      dropout=0.1)
        output = attention(queries, keys, values, valid_lens, training=False)
        # output.shape
        self.assertTrue(output.shape==[2, 1, 4])

    def test_MutiHeadAttention(self):
        """
        query and key have the same length
        Returns:

        """
        num_hiddens, num_heads = 100, 5
        query_size = num_hiddens
        key_size = num_hiddens
        attention = MultiHeadAttention(
            # num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)

        batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, tf.constant([
            3, 2])
        X = tf.ones((batch_size, num_queries, query_size))
        Y = tf.ones((batch_size, num_kvpairs, key_size))
        res = attention(queries=X, keys=Y, values=Y, valid_lens=valid_lens, training=False)
        self.assertTrue(res.shape == [2, 4, 100])

