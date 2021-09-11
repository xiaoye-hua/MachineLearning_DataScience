# -*- coding: utf-8 -*-
# @File    : test_NNModel.py
# @Author  : Hua Guo
# @Time    : 2020/2/5 下午5:04
# @Disc    :
import numpy as np
from unittest import TestCase

from src.Model.NN.NNModel import NNModel
from Unittest.test_Model.test_NN.testCases import L_model_backward_test_case, print_grads, linear_backward_test_case
from Unittest.test_Model.test_NN.testCases import linear_activation_backward_test_case
from Unittest.test_Model.test_NN.testCases import (
    compute_cost_test_case, update_parameters_test_case,
            compute_cost_with_regularization_test_case,
    backward_propagation_with_regularization_test_case,
    backward_propagation_with_dropout_test_case,
    forward_propagation_with_dropout_test_case
)


class TestNNModel(TestCase):
    def setUp(self) -> None:
        self.model = NNModel([3, 1])

    def test_cost_function(self):
        Y, AL = compute_cost_test_case()
        result = self.model._compute_cost(AL=AL, Y=Y)
        print(result)
        self.assertTrue(
            result == 0.41493159961539694
        )

    def test_cost_with_reguraztion(self):
        model = NNModel([9])
        A3, Y_assess, model.model_params = compute_cost_with_regularization_test_case()
        cost = model.compute_cost_with_regularization(
            A3=A3,
            Y=Y_assess,
            parameters=model.model_params,
            lambd=0.1
        )
        print(cost)
        # print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd=0.1)))

    def test_back_prob(self):
        AL, Y_assess, caches = L_model_backward_test_case()
        # print(AL)
        # print(Y_assess)
        # print(caches)
        # print()
        grads = self.model._back_prob(AL, Y_assess, caches)
        print_grads(grads)

    def test_back_prob_with_lambd(self):
        # X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
        AL, Y_assess, caches = L_model_backward_test_case()
        grads = self.model._back_prob(
            # X_assess,
            AL,
                                      Y_assess,
                                      caches
                                      , lambd=0.7
                                      )
        print("dW1 = " + str(grads["dW1"]))
        print("dW2 = " + str(grads["dW2"]))
        # print("dW3 = " + str(grads["dW3"]))

    # def test_back_prob_with_dropout(self):
    #     X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
    #
    #     gradients = self.model._back_prob(X_assess, Y_assess, cache, dropout_keep_prob=0.8)
    #
    #     print("dA1 = " + str(gradients["dA1"]))
    #     print("dA2 = " + str(gradients["dA2"]))

    def test_forward_prob_dropout(self):
        """
        A3 = [[3.69747206e-01 2.93815585e-04 4.96833893e-01 1.44689281e-02
        4.96833893e-01]]
        :return:
        """
        model = NNModel([1, 2])
        X_assess, model.model_params = forward_propagation_with_dropout_test_case()

        A3, cache = model._forward_prob(X_assess, dropout_keep_prob=0.7)
        print("A3 = " + str(A3))

    def test_linear_backwards(self):
        dZ, linear_cache = linear_backward_test_case()

        dA_prev, dW, db = self.model._linear_backward(dZ, linear_cache)
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))

    def test_linear_activation_backward_relu(self):
        AL, linear_activation_cache = linear_activation_backward_test_case()
        dA_prev, dW, db = self.model._linear_activation_backward(AL, linear_activation_cache, activation="relu")
        print("relu:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))
        #

    def test_linear_activation_backward_sigmoid(self):
        AL, linear_activation_cache = linear_activation_backward_test_case()
        dA_prev, dW, db = self.model._linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
        print("sigmoid:")
        print("dA_prev = " + str(dA_prev))
        print("dW = " + str(dW))
        print("db = " + str(db))
        # expect = np.array(
        #     [[0.10266786, 0.09778551, -0.01968084]]
        # )
        # self.assertTrue(
        #     np.array_equal(dW, expect)
        # )

    def test_update_parameters(self):
        model = NNModel([0])
        model.model_params, grads = update_parameters_test_case()

        model._update_params(grads, 0.1)

        print("W1 = " + str(model.model_params["W1"]))
        print("b1 = " + str(model.model_params["b1"]))
        print("W2 = " + str(model.model_params["W2"]))
        print("b2 = " + str(model.model_params["b2"]))