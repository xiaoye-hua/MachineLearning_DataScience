# -*- coding: utf-8 -*-
# @File    : Transformer.py
# @Author  : Hua Guo
# @Time    : 2021/8/31 上午11:35
# @Disc    :
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, LayerNormalization, Dense, Activation, Embedding

from src.BaseClass.DLModel import DLModel
from src.Model.SequenceModel.Attention import MultiHeadAttention


class PositionalEncoding(DLModel):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        """

        Args:
            num_hiddens: dimension of position encoding
            dropout: dropout rate
            max_len: init max_len of position encoding
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(rate=dropout)
        self.P = np.zeros(shape=(1, max_len, num_hiddens))
        X = np.arange(0, max_len, 1).reshape(-1, 1)/np.power(10000, np.arange(0, num_hiddens, 2))
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, training=None, mask=None):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)


class AddNorm(DLModel):
    def __init__(self, norm_axis, dropout):
        """
        Args:
            norm_axis: axis pramter for LayerNormalization; inter or list
            dropout: dropout rate
        """
        super(AddNorm, self).__init__()
        self.droupout = Dropout(rate=dropout)
        self.layNorm = LayerNormalization(axis=norm_axis)

    def call(self, X, Y, training=None, mask=None):
        return self.layNorm(self.droupout(Y) + X)


class PositionWiseFFN(DLModel):
    def __init__(self, num_hiddens, num_output):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = Dense(units=num_hiddens)
        self.ac = Activation(activation='relu')
        self.dense2 = Dense(units=num_output)

    def call(self, inputs, training=None, mask=None):
        return self.dense2(self.ac(self.dense1(inputs)))


class EncoderBlock(DLModel):
    def __init__(self, num_hidden: int, num_head: int, norm_shape: list, ffn_num_hidden: int, dropout: float, bias=False) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_hidden=num_hidden, num_head=num_head, dropout=dropout, bias=bias)
        self.add_norm1 = AddNorm(norm_axis=norm_shape, dropout=dropout)
        self.ffn = PositionWiseFFN(num_hiddens=ffn_num_hidden, num_output=num_hidden)
        self.add_norm2 = AddNorm(norm_axis=norm_shape, dropout=dropout)

    def call(self, inputs, valid_len, training=None, mask=None):
        X = self.add_norm1(inputs, self.attention(queries=inputs, keys=inputs, values=inputs, valid_lens=valid_len))
        res = self.add_norm1(X, self.ffn(X))
        return res


class TransformerEncoder(DLModel):
    def __init__(self, vocab_size: int, encoder_block_num: int, num_hidden: int, num_head: int, norm_shape: list, ffn_num_hidden: int, dropout: float, bias=False):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=num_hidden)
        self.positional_encode = PositionalEncoding(num_hiddens=num_hidden, dropout=dropout)
        self.blocks = [EncoderBlock(num_hidden=num_hidden, num_head=num_head, norm_shape=norm_shape, ffn_num_hidden=ffn_num_hidden, bias=bias, dropout=dropout) for _ in range(encoder_block_num)]

    def call(self, inputs, vaid_lens, training=None, mask=None):
        X = self.positional_encode(self.embedding(inputs))
        for blk in self.blocks:
            X = blk(inputs=X, valid_len=vaid_lens)
        return X


class DecoderBlock(DLModel):
    def __init__(self, num_hidden, num_head, norm_shape, ffn_num_hidden, droupout, i, bias=False):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hidden=num_hidden, num_head=num_head, dropout=droupout, bias=bias)
        self.add_norm1 = AddNorm(norm_axis=norm_shape, dropout=droupout)
        self.attention2 = MultiHeadAttention(num_hidden=num_hidden, num_head=num_head, dropout=droupout, bias=bias)
        self.add_norm2 = AddNorm(norm_axis=norm_shape, dropout=droupout)
        self.ffn = PositionWiseFFN(num_hiddens=num_hidden, num_output=ffn_num_hidden)
        self.add_norm3 = AddNorm(norm_axis=norm_shape, dropout=droupout)

    def call(self, inputs, state, training=None, mask=None):
        """

        Args:
            inputs:
            state: [encoder_output, valid_len, hidden_state]
                hidden_state[self.i](will be used as the queries of encoder-decoder attention): representation of the decoder output
                    1. at the ithe block
                    2. to the current timestep

            training:
            mask:

        Returns:

        """
        encoder_output, enc_valid_len = state[0], state[1]
        if state[2][self.i] is None:
            key_values = inputs
        else:
            key_values = tf.concat([state[2][self.i], inputs], axis=1)
        state[2][self.i] = key_values # --> will be used for next timestep
        if training:
            batch_size, num_steps, _ = inputs.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)),
                repeats=batch_size, axis=0)

        else:
            dec_valid_lens = None
        X = self.attention1(queries=inputs, keys=key_values, values=key_values, valid_lens=enc_valid_len)
        X1 = self.add_norm1(inputs, X)
        X2 = self.attention2(queries=X1, keys=encoder_output, values=encoder_output, valid_lens=dec_valid_lens)
        X2 = self.add_norm2(X1, X2)
        res = self.add_norm3(X2, self.ffn(X2))
        return res, state








