# -*- coding: utf-8 -*-
# @File    : KNN.py
# @Author  : Hua Guo
# @Time    : 2021/9/12 下午11:15
# @Disc    :
from collections import Counter
from math import sqrt


def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance +=(point1[i] - point2[i]) ** 2
    return sqrt(distance)


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def KNN(training_data, target, k, func):
    """
    training_data: all training data point
    target: new point
    k: user-defined constant, number of closest training data
    func: functions used to get the the target label
    """
    # Step one: calculate the Euclidean distance between the new point and all training data
    neighbors= []
    for index, data in enumerate(training_data):
        # distance between the target data and the current example from the data.
        distance = euclidean_distance(data[:-1], target)
        neighbors.append((distance, index))

    # Step two: pick the top-K closest training data
    sorted_neighbors = sorted(neighbors)
    k_nearest = sorted_neighbors[:k]
    k_nearest_labels = [training_data[i][1] for distance, i in k_nearest]

    # Step three: For regression problem, take the average of the labels as the result;
    #             for classification problem, take the most common label of these labels as the result.
    return k_nearest, func(k_nearest_labels)