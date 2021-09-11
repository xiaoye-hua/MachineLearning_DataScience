# -*- coding: utf-8 -*-
# @File    : test_Transformer.py
# @Author  : Hua Guo
# @Time    : 2021/8/31 上午11:44
# @Disc    :
import tensorflow as tf
import numpy as np
from unittest import TestCase

# from src.TODO.d2l import PositionalEncoding, EncoderBlock, AddNorm, PositionWiseFFN, TransformerEncoder, DecoderBlock
from src.Model.SequenceModel.Transformer import PositionalEncoding, AddNorm, PositionWiseFFN, EncoderBlock, TransformerEncoder, DecoderBlock
import src.utils.d2l as d2l


class TestTransformer(TestCase):
    def setUp(self) -> None:
        self.X = tf.ones((2, 100, 24))
        self.valid_len = tf.constant([3, 2])

    def test_PositionalEncoding(self):
        encoding_dim, num_steps = 32, 60
        pos_encoding = PositionalEncoding(num_hiddens=encoding_dim, dropout=0)
        X = tf.zeros((1, num_steps, encoding_dim))
        X = pos_encoding(X=X, training=False)
        P = pos_encoding.P[:, :X.shape[1], :]
        self.assertTrue(X.shape == [1, num_steps, encoding_dim])
        self.assertTrue(P.shape == [1, num_steps, encoding_dim])

    def test_AddNorm(self):
        add_norm = AddNorm(
            [1, 2],
            0.5)  # Normalized_shape is: [i for i in range(len(input.shape))][1:]
        res = add_norm(tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), training=False)
        self.assertTrue(res.shape == (2, 3, 4))

    def test_PositionWiseFFN(self):
        ffn = PositionWiseFFN(4, 8)
        res = ffn(tf.ones((2, 3, 4)))
        self.assertTrue(res.shape == (2, 3, 8))

    def test_EncoderBlock(self):
        norm_shape = [i for i in range(len(self.X.shape))][1:]
        encoder_blk = EncoderBlock(
            # 24, 24, 24,
                                   num_hidden=24, norm_shape=norm_shape, ffn_num_hidden=48, num_head=8, dropout=0.5)
        res = encoder_blk(self.X, self.valid_lens, training=False)
        self.assertTrue(res.shape == (2, 100, 24))

    def test_TransformerEncoder(self):
        valid_lens = tf.constant([3, 2])
        encoder = TransformerEncoder(vocab_size=200, num_hidden=24,
                                     # 24, 24, 24,
                                     norm_shape=[1, 2], ffn_num_hidden=48, num_head=8, encoder_block_num=2, dropout=0.5)
        res = encoder(tf.ones((2, 100)), valid_lens, training=False)
        self.assertTrue(res.shape == (2, 100, 24))

    def test_DecoderBlock(self):
        decoder_blk = DecoderBlock(
            # key_size=24, query_size=24, value_size=24,
                                   num_hidden=24, norm_shape=[1, 2],
                                   ffn_num_hidden=48, num_head=8,
                                   droupout=0.5,
                                   i=0
                                   )
        X = tf.ones((2, 100, 24))
        encoder_output = tf.ones((2, 100, 24))
        # passed from decoder.init_state()
        state = [encoder_output, self.valid_len, [None]]
        res = decoder_blk(X, state, training=False)[0]
        self.assertTrue(res.shape == (2, 100, 24))

    def test_TransformerDecoder(self):
        pass


    def test_Transformer(self):
        pass
        # num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        # lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
        # ffn_num_hiddens, num_heads = 64, 4
        # key_size, query_size, value_size = 32, 32, 32
        # norm_shape = [2]
        #
        # train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
        # encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
        #                              num_hiddens, norm_shape, ffn_num_hiddens,
        #                              num_heads, num_layers, dropout)
        # decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
        #                              num_hiddens, norm_shape, ffn_num_hiddens,
        #                              num_heads, num_layers, dropout)
        # net = d2l.EncoderDecoder(encoder, decoder)
        # d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)