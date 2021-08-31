# -*- coding: utf-8 -*-
# @File    : Seq2SeqAttention.py
# @Author  : Hua Guo
# @Time    : 2021/8/30 下午9:03
# @Disc    :
import tensorflow as tf
from tensorflow.keras import layers

from src.utils import d2l as d2l
from src.utils.d2l import AttentionDecoder
from src.BaseClass.DLModel import DLModel
from src.Model.SequenceModel.Attention import AdditiveAttention


class Seq2SeqAttentionDecoder(DLModel):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
        self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.rnn = layers.RNN(
            layers.StackedRNNCells([layers.GRUCell(units=num_hiddens, dropout=dropout) for _ in range(num_layers)])
            , return_state=True
            , return_sequences=True
        )
        self.dense = layers.Dense(units=vocab_size)

    def call(self, inputs, state, training=None, mask=None):
        X = self.embed(inputs)
        enc_output, hidden_state, valid_len = state
        # ================================
        # code for Seq2Seq without attention
        # ================================
        # state = tf.repeat(tf.expand_dims(enc_hidden[-1], axis=1), repeats=X.shape[1], axis=1)
        # x_state_concat = tf.concat([X, state], axis=-1)
        # rnn_out = self.rnn(x_state_concat)
        # ================================
        # code for Seq2Seq with attention
        # ================================
        # dimension: [batch, timestep, size] -> [timestep, batch, size]
        X = tf.transpose(X, [1, 0, 2])
        output_lst, self.attention_paramters = [], []
        # hidden = [-1]
        for idx, x in enumerate(X):
            query = tf.expand_dims(hidden_state[-1], axis=1)
            context = self.attention(query, enc_output, enc_output, valid_len)
            x_context_concat = tf.concat([tf.expand_dims(x, axis=1), context], axis=-1)
            # print("+"*20)
            # print(idx)
            # print(f"x_context: {x_context_concat.shape}")
            # print(f"hidden state: {hidden_state[0].shape}")
            output = self.rnn(x_context_concat, hidden_state)
            output_lst.append(output[0])
            self.attention_paramters.append(context)
            hidden_state = output[1:]

        output = self.dense(tf.concat(output_lst, axis=1))
        return output, [enc_output, hidden_state, valid_len]

