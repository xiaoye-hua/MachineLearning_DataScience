# -*- coding: utf-8 -*-
# @File    : test_Seq2SeqAttention.py
# @Author  : Hua Guo
# @Time    : 2021/8/30 下午9:05
# @Disc    :
import tensorflow as tf
from unittest import TestCase

from src.Model.SequenceModel.Seq2SeqAttention import Seq2SeqAttentionDecoder


class TestSeq2SeqAttention(TestCase):

    def test_mydecoder(self):
        decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                          num_layers=2)
        X = tf.zeros((4, 7))
        state = (tf.random.uniform(shape=(4, 7, 16))
                 ,[tf.random.uniform(shape=(4, 16)), tf.random.uniform(shape=(4, 16))]
                 , None
                 )
        output, state = decoder(X, state, training=False)
        output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape

        # outputs, [enc_outputs, hidden_state, enc_valid_lens]
        self.assertTrue(output.shape == [4, 7, 10])
        self.assertTrue(state[0].shape == [4, 7, 16])
        self.assertTrue(state[1][0].shape == [4, 16], msg=state[1][0].shape)
