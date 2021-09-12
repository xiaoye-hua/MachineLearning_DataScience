# -*- coding: utf-8 -*-
# @File    : RNN.py
# @Author  : Hua Guo
# @Time    : 2021/9/12 下午11:08
# @Disc    :
from typing import List

import numpy as np
from scipy.special import softmax


class RNN:
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
        Waa = np.random.normal(scale=scale, size=[self.hidden_dim, self.hidden_dim])
        Wax = np.random.normal(scale=scale, size=[self.hidden_dim, self.input_dim])
        Wy = np.random.normal(scale=scale, size=[self.out_dim, self.hidden_dim])
        ba = np.zeros(shape=[self.hidden_dim, 1])
        by = np.zeros(shape=[self.out_dim, 1])
        return [Waa, Wax, Wy, ba, by]

    def _init_hidden_state(self) -> np.array:
        return np.zeros(shape=[self.hidden_dim, self.batch_size])

    def forward(self, input_vector: np.array) -> np.array:
        """
        input_vector:
            dimension: [num_steps, self.input_dim, self.batch_size]
        out_vector:
            dimension: [num_steps, self.output_dim, self.batch_size]
        """
        Waa, Wax, Wy, ba, by = self.params
        output_vector = []
        for vector in input_vector:
            self.hidden_state = np.tanh(
                np.dot(Waa, self.hidden_state) + np.dot(Wax, vector) + ba
            )
            y = softmax(
                np.dot(Wy, self.hidden_state) + by
            )
            output_vector.append(y)
        return np.array(output_vector)