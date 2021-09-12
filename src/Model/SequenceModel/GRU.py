# -*- coding: utf-8 -*-
# @File    : GRU.py
# @Author  : Hua Guo
# @Time    : 2021/9/12 下午11:08
# @Disc    :
from typing import List

import numpy as np
from scipy.special import expit, softmax

from src.Model.SequenceModel.LSTM import input_dim


class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, batch_size=1) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim
        self.batch_size = batch_size
        # initialization
        self.params = self._init_params()
        self.hidden_state = self._init_hidden_state()

    def _init_params(self) -> List[np.array]:
        scale = 0.01
        def param_single_layer():
            w = np.random.normal(scale=scale, size=(self.hidden_dim, self.hidden_dim+input_dim))
            b = np.zeros(shape=[self.hidden_dim, 1])
            return w, b

        # reset, update gate
        Wr, br = param_single_layer()
        Wu, bu = param_single_layer()
        # output layer
        Wy = np.random.normal(scale=scale, size=[self.out_dim, self.hidden_dim])
        by = np.zeros(shape=[self.out_dim, 1])
        return [Wr, br, Wu, bu, Wy, by]

    def _init_hidden_state(self) -> np.array:
        return np.zeros(shape=[self.hidden_dim, self.batch_size])

    def forward(self, input_vector: np.array) -> np.array:
        """
        input_vector:
            dimension: [num_steps, self.input_dim, self.batch_size]
        out_vector:
            dimension: [num_steps, self.output_dim, self.batch_size]
        """
        Wr, br, Wu, bu, Wy, by = self.params
        output_vector = []
        for vector in input_vector:
            # expit in scipy is sigmoid function
            reset_gate = expit(
                np.dot(Wr, np.concatenate([self.hidden_state, vector], axis=0)) + br
            )
            update_gate = expit(
                np.dot(Wu, np.concatenate([self.hidden_state, vector], axis=0)) + bu
            )
            candidate_hidden = np.tanh(
                reset_gate * self.hidden_state
            )
            self.hidden_state = update_gate * self.hidden_state + (1-update_gate) * candidate_hidden
            y = softmax(
                np.dot(Wy, self.hidden_state) + by
            )
            output_vector.append(y)
        return np.array(output_vector)