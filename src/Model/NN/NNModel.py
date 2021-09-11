# -*- coding: utf-8 -*-
# @File    : NNModel.py
# @Author  : Hua Guo
# @Time    : 2020/2/3 下午6:28
# @Disc    :
import numpy as np
import matplotlib.pyplot as plt
from typing import Type, List

from src.Model.NN.AbstractClass import BaseModel
from src.Model.NN.Data import load_catnocat_dataset
from src.Model.NN.ParamsInitializer import ParamsInitializer
from src.Model.NN.ActivationForward import ActivationForward
from src.Model.NN.ActivationBackward import ActivationBackward


class NNModel(BaseModel):
    """
    N-layer full connected neural network model

    General methodology:
        1. Initialize parameters / Define hyperparameters
        2. Loop for num_iterations:
            a. Forward propagation
            b. Compute cost function
            c. Backward propagation
            d. Update parameters (using parameters, and grads from backprop)
        4. Use trained parameters to predict labels
    """
    def __init__(self, layer_dims: List[int]) -> None:
        # super(NNModel, self).__init__()
        self.layer_dims = layer_dims
        # self.layer_nums = len(self.layer_dims) - 1
        self.model_params = {}
        self.cost_lst =[]
        self.cost_save_interval = 100

    def train(self, X: Type[np.array], Y: Type[np.array],
              learning_rate: float, num_iterations: int, print_cost=False,
              random_seed=None,
              lambd=None,
              dropout_keep_prob=None
              ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self._initialize_params()
        for i in range(num_iterations):
            AL, caches = self._forward_prob(X, dropout_keep_prob=dropout_keep_prob)
            if lambd is not None or lambd != 0:
                cost = self.compute_cost_with_regularization(
                    A3=AL,
                    Y=Y,
                    parameters=model.model_params,
                    lambd=lambd
                )
            else:
                cost = self._compute_cost(AL, Y)
            grads = self._back_prob(AL, Y, caches, lambd=lambd, dropout_keep_prob=dropout_keep_prob)
            self._update_params(grads=grads, learning_rate=learning_rate)
            if i%self.cost_save_interval ==0 :
                self.cost_lst.append(cost)
            if print_cost and i%self.cost_save_interval == 0:
                print(f"Cost after {i+1} iterations: {cost}")
        self._plot_cost(learning_rate=learning_rate)

    def predict(self, X: Type[np.array]) -> Type[np.array]:
        prob, _ = self._forward_prob(X)
        return prob

    def evaluation(self, X, Y):
        m = X.shape[1]
        p, _ = self._forward_prob(X)
        for i in range(0, p.shape[1]):
            if p[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        accuracy = np.sum([p[0, i] == Y[0, i] for i in range(0, p.shape[1])]) / float(m)
        print(f"Total Sample: {m}")
        print(f"Accuracy: {accuracy}")

    def _plot_cost(self, learning_rate):
        if len(self.cost_lst) > 0:
            plt.plot(self.cost_lst)
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

    def _initialize_params(self) -> None:
        for idx in range(1, len(self.layer_dims)):
            self.model_params["W" + str(idx)] = ParamsInitializer.Xavier_initialization(layer_dims[idx], layer_dims[idx - 1])
            self.model_params["b" + str(idx)] = ParamsInitializer.zero_initialization(layer_dims[idx], 1)

    def _forward_prob(self, X, dropout_keep_prob=None):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:i
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(self.model_params) // 2  # number of layers in the neural network
        for l in range(1, L):
            A_prev = A
            W = self.model_params['W' + str(l)]
            b = self.model_params['b' + str(l)]
            A, cache = self._linear_activation_forward(A_prev, W, b, 'relu', dropout_keep_prob=dropout_keep_prob)
            caches.append(cache)
        W = self.model_params['W' + str(L)]
        b = self.model_params['b' + str(L)]
        A_prev = A
        AL, cache = self._linear_activation_forward(A_prev, W, b, 'sigmoid')
        caches.append(cache)
        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def _back_prob(self, AL, Y, caches, lambd=None, dropout_keep_prob=None) -> dict:
        grads = {}
        L = len(caches) # the number of layers
        # initialize diravitive of cost with respect to AL
        dA = -(np.divide(Y, AL) - np.divide((1-Y), (1-AL)))
        # last layer with sigmoid backward
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW"+ str(L)], grads["db" + str(L)] = self._linear_activation_backward(
            dA=dA,
            cache=current_cache,
            activation="sigmoid",
            lambd=lambd,
        )
        # rest layers backward with relu activation
        for idx in reversed(range(L-1)):
            current_cache = caches[idx]
            grads["dA" + str(idx+1)], grads["dW" + str(idx+1)], grads["db" + str(idx+1)] = self._linear_activation_backward(
                dA=grads["dA" + str(idx+2)],
                cache=current_cache,
                activation="relu",
                lambd=lambd,
                dropout_keep_prob=dropout_keep_prob
            )
        return grads

    def _update_params(self, grads: dict, learning_rate: float) -> None:
        for key, value in self.model_params.items():
            self.model_params[key] = value - learning_rate*grads["d" + key]

    def _linear_activation_forward(self, A_prev, W, b, activation, dropout_keep_prob=None) -> tuple:
        """

        :param A_prev:
        :param W:
        :param b:
        :param activation:
        :param dropout_keep_prob:
        :return:
            A:
            caches:
                linear_caches:
                    A_prev
                    W
                    b
                activation_caches
                    Z
                    D
        """
        activation_dict = {
            "relu": ActivationForward.relu,
            "sigmoid": ActivationForward.sigmoid
        }
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        A, activation_cache = activation_dict[activation](Z)
        if dropout_keep_prob is not None:
            # np.random.seed(1)
            D1 = np.random.rand(A.shape[0], A.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
            D1 = (D1 < dropout_keep_prob)  # Step 2: convert entries of D1 to 0 or 1 (using dropout_keep_prob as the threshold)
            A = A * D1  # Step 3: shut down some neurons of A
            A = A / dropout_keep_prob
            activation_cache = [activation_cache]
            activation_cache.append(D1)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache

    def _linear_activation_backward(self, dA, cache, activation: str, lambd=None, dropout_keep_prob=None):
        """
        :param dA:
        :param cache:
        :param activation:
        :return:
        """
        activation_back_dict = {
            "relu": ActivationBackward.relu_backward,
            "sigmoid": ActivationBackward.sigmoid_backward
        }
        # get linear & activation cache
        linear_caches, activation_caches = cache
        # get dZ
        dZ = activation_back_dict[activation](dA=dA, cache=activation_caches)
        # get dA_prev, dw, db
        dA_prev, dw, db = self._linear_backward(dZ=dZ, cache=linear_caches, lambd=lambd)
        if dropout_keep_prob is not None:
            D = activation_caches[-1]
            dA_prev = dA_prev * D  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
            dA_prev = dA_prev / dropout_keep_prob
        return dA_prev, dw, db

    def _linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def _linear_backward(self, dZ, cache, lambd=None):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1. / m * np.dot(dZ, cache[0].T)
        if lambd is not None:
            dW += lambd / m * W
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[1].T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def _compute_cost(self, AL, Y):
        """

        :param AL:
        :param Y: shape [1, ele_nums]
        :return:
        """
        m = Y.shape[1]
        prob = Y * np.log(AL) + (1-Y) * np.log(1-AL)
        cost = -1/m * np.sum(prob)
        cost = np.squeeze(cost)
        return cost

    def compute_cost_with_regularization(self, A3, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization. See formula (2) above.

        Arguments:
        A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model

        Returns:
        cost - value of the regularized loss function (formula (2))
        """
        m = Y.shape[1]
        L = len(parameters) // 2
        cross_entropy_cost = self._compute_cost(A3, Y)  # This gives you the cross-entropy part of the cost
        L2_regularization_cost = 0
        for idx in range(L):
            W = parameters["W" + str(idx+1)]
            L2_regularization_cost += 1. / m * lambd / 2. * np.sum(np.square(W))
        cost = cross_entropy_cost + L2_regularization_cost
        return cost


if __name__ == '__main__':
    """
    when random_seed is set to be 1:
    
        Cost after 1 iterations: 0.8288794776720926
        Cost after 101 iterations: 0.5865829684519076
        Cost after 201 iterations: 0.4679368422445453
        Cost after 301 iterations: 0.3764388409539531
        Cost after 401 iterations: 0.3318641943960499
        Cost after 501 iterations: 0.3035289711874395
        Cost after 601 iterations: 0.280052091025139
        Cost after 701 iterations: 0.260164802374736
        Cost after 801 iterations: 0.2430318152673041
        Cost after 901 iterations: 0.22807403044483376
        Total Sample: 209
        Accuracy: 0.9712918660287081
        Total Sample: 50
        Accuracy: 0.74
    """
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_catnocat_dataset()
    layer_dims = [12288, 1]
    num_iter = 1000
    learning_rate = 0.005
    lambd = 0.1
    dropout_keep_prob = 0.7
    random_seed = 1
    model = NNModel(
        layer_dims=layer_dims
                    )
    model.train(
        X=train_set_x,
        Y=train_set_y,
        num_iterations=num_iter,
        learning_rate=learning_rate,
        print_cost=True,
        random_seed=random_seed,
        lambd=lambd,
        dropout_keep_prob=dropout_keep_prob
    )
    # result = model.predict(test_set_x)
    model.evaluation(
        X=train_set_x,
        Y=train_set_y
    )
    model.evaluation(
        X=test_set_x,
        Y=test_set_y
    )

