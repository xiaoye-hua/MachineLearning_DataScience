# -*- coding: utf-8 -*-
# @File    : LSTM.py
# @Author  : Hua Guo
# @Time    : 2020/3/17 下午8:29
# @Disc    :
import numpy as np
from scipy.special import softmax
from scipy.special import expit
from typing import List


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, batch_size=1) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim
        self.batch_size = batch_size
        # initialization
        self.params = self._init_params()
        self.hidden_state = self._init_hidden_state()
        self.memory_state = self._init_hidden_state()

    def _init_params(self) -> List[np.array]:
        scale = 0.01
        def param_single_layer():
            w = np.random.normal(scale=scale, size=(self.hidden_dim, self.hidden_dim+input_dim))
            b = np.zeros(shape=[self.hidden_dim, 1])
            return w, b

        # forget, input, output gate + candidate memory state
        Wf, bf = param_single_layer()
        Wi, bi = param_single_layer()
        Wo, bo = param_single_layer()
        Wc, bc = param_single_layer()
        # output layer
        Wy = np.random.normal(scale=scale, size=[self.out_dim, self.hidden_dim])
        by = np.zeros(shape=[self.out_dim, 1])
        return [Wf, bf, Wi, bi, Wo, bo, Wc, bc, Wy, by]

    def _init_hidden_state(self) -> np.array:
        return np.zeros(shape=[self.hidden_dim, self.batch_size])

    def forward(self, input_vector: np.array) -> np.array:
        """
        input_vector:
            dimension: [num_steps, self.input_dim, self.batch_size]
        out_vector:
            dimension: [num_steps, self.output_dim, self.batch_size]
        """
        Wf, bf, Wi, bi, Wo, bo, Wc, bc, Wy, by = self.params
        output_vector = []
        for vector in input_vector:
            # expit in scipy is sigmoid function
            foget_gate = expit(
                np.dot(Wf, np.concatenate([self.hidden_state, vector], axis=0)) + bf
            )
            input_gate = expit(
                np.dot(Wi, np.concatenate([self.hidden_state, vector], axis=0)) + bi
            )
            output_gate = expit(
                np.dot(Wo, np.concatenate([self.hidden_state, vector], axis=0)) + bo
            )
            candidate_memory = np.tanh(
                np.dot(Wc, np.concatenate([self.hidden_state, vector], axis=0)) + bc
            )
            self.memory_state = foget_gate * self.memory_state + input_gate * candidate_memory
            self.hidden_state = output_gate * np.tanh(self.memory_state)
            y = softmax(
                np.dot(Wy, self.hidden_state) + by
            )
            output_vector.append(y)
        return np.array(output_vector)


if __name__ == "__main__":
    input_data = np.array([
        [
            [1, 3]
            , [2, 4]
            , [3, 6]
        ]
        , [
            [4, 3]
            , [3, 4]
            , [1, 5]
        ]
    ])
    batch_size = 2
    input_dim = 3
    output_dim = 4
    hidden_dim = 5
    time_step = 2
    # rnn = RNN(input_dim=input_dim, batch_size=batch_size, output_dim=output_dim, hidden_dim=hidden_dim)
    # output_vector = rnn.forward(input_vector=input_data)
    # print("RNN:")
    # print(f"Input data dimensions: {input_data.shape}")
    # print(f"Output data dimensions {output_vector.shape}")
    rnn = LSTM(input_dim=input_dim, batch_size=batch_size, output_dim=output_dim, hidden_dim=hidden_dim)
    output_vector = rnn.forward(input_vector=input_data)
    print("LSTM:")
    print(f"Input data dimensions: {input_data.shape}")
    print(f"Output data dimensions {output_vector.shape}")