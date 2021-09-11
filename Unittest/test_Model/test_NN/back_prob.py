# # -*- coding: utf-8 -*-
# # @File    : back_prob.py
# # @Author  : Hua Guo
# # @Time    : 2020/2/7 下午3:44
# # @Disc    :
# import numpy as np
# from Unittest.test_Model.test_NN.testCases_v3 import L_model_backward_test_case, print_grads
#
#
# def relu_backward(dA, cache):
#     """
#     Implement the backward propagation for a single RELU unit.
#
#     Arguments:
#     dA -- post-activation gradient, of any shape
#     cache -- 'Z' where we store for computing backward propagation efficiently
#
#     Returns:
#     dZ -- Gradient of the cost with respect to Z
#     """
#
#     Z = cache
#     dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
#
#     # When z <= 0, you should set dz to 0 as well.
#     dZ[Z <= 0] = 0
#
#     assert (dZ.shape == Z.shape)
#
#     return dZ
#
#
# def sigmoid_backward(dA, cache):
#     """
#     Implement the backward propagation for a single SIGMOID unit.
#
#     Arguments:
#     dA -- post-activation gradient, of any shape
#     cache -- 'Z' where we store for computing backward propagation efficiently
#
#     Returns:
#     dZ -- Gradient of the cost with respect to Z
#     """
#
#     Z = cache
#
#     s = 1 / (1 + np.exp(-Z))
#     dZ = dA * s * (1 - s)
#
#     assert (dZ.shape == Z.shape)
#
#     return dZ
#
# # GRADED FUNCTION: linear_backward
#
# def linear_backward(dZ, cache):
#     """
#     Implement the linear portion of backward propagation for a single layer (layer l)
#
#     Arguments:
#     dZ -- Gradient of the cost with respect to the linear output (of current layer l)
#     cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
#
#     Returns:
#     dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
#     dW -- Gradient of the cost with respect to W (current layer l), same shape as W
#     db -- Gradient of the cost with respect to b (current layer l), same shape as b
#     """
#     A_prev, W, b = cache
#     m = A_prev.shape[1]
#
#     ### START CODE HERE ### (≈ 3 lines of code)
#     dW = 1. / m * np.dot(dZ, cache[0].T)
#     db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
#     dA_prev = np.dot(cache[1].T, dZ)
#     ### END CODE HERE ###1
#
#     assert (dA_prev.shape == A_prev.shape)
#     assert (dW.shape == W.shape)
#     assert (db.shape == b.shape)
#
#     return dA_prev, dW, db
# # GRADED FUNCTION: linear_activation_backward
#
# def linear_activation_backward(dA, cache, activation):
#     """
#     Implement the backward propagation for the LINEAR->ACTIVATION layer.
#
#     Arguments:
#     dA -- post-activation gradient for current layer l
#     cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
#     activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
#
#     Returns:
#     dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
#     dW -- Gradient of the cost with respect to W (current layer l), same shape as W
#     db -- Gradient of the cost with respect to b (current layer l), same shape as b
#     """
#     linear_cache, activation_cache = cache
#
#     if activation == "relu":
#         ### START CODE HERE ### (≈ 2 lines of code)
#         dZ = relu_backward(dA, activation_cache)
#         dA_prev, dW, db = linear_backward(dZ, linear_cache)
#         ### END CODE HERE ###
#
#     elif activation == "sigmoid":
#         ### START CODE HERE ### (≈ 2 lines of code)
#         dZ = sigmoid_backward(dA, activation_cache)
#         dA_prev, dW, db = linear_backward(dZ, linear_cache)
#         ### END CODE HERE ###
#
#     return dA_prev, dW, db
#
#
# # GRADED FUNCTION: L_model_backward
#
# def L_model_backward(AL, Y, caches):
#     """
#     Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
#
#     Arguments:
#     AL -- probability vector, output of the forward propagation (L_model_forward())
#     Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
#     caches -- list of caches containing:
#                 every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
#                 the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
#
#     Returns:
#     grads -- A dictionary with the gradients
#              grads["dA" + str(l)] = ...
#              grads["dW" + str(l)] = ...
#              grads["db" + str(l)] = ...
#     """
#     grads = {}
#     L = len(caches)  # the number of layers
#     m = AL.shape[1]
#     Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
#
#     # Initializing the backpropagation
#     ### START CODE HERE ### (1 line of code)
#     dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
#     ### END CODE HERE ###
#
#     # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
#     ### START CODE HERE ### (approx. 2 lines)
#     current_cache = caches[L - 1]
#     grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
#                                                                                                   'sigmoid')
#     ### END CODE HERE ###
#
#     for l in reversed(range(L - 1)):
#         # lth layer: (RELU -> LINEAR) gradients.
#         # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
#         ### START CODE HERE ### (approx. 5 lines)
#         current_cache = caches[l]
#         dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, 'relu')
#         grads["dA" + str(l + 1)] = dA_prev_temp
#         grads["dW" + str(l + 1)] = dW_temp
#         grads["db" + str(l + 1)] = db_temp
#         ### END CODE HERE ###
#
#     return grads
#
#
# if __name__ == "__main__":
#
#     AL, Y_assess, caches = L_model_backward_test_case()
#     print(AL)
#     print(Y_assess)
#     print(caches)
#     grads = L_model_backward(AL, Y_assess, caches)
#     print_grads(grads)