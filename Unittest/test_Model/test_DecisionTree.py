# -*- coding: utf-8 -*-
# @File    : test_DecisionTree.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午9:01
# @Disc    :
import numpy as np
from unittest import TestCase
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.Model.DecisionTree.CART import CART
from src.Model.DecisionTree.ID3DecisionTree import ID3DecisionTree
from src.Model.DecisionTree.C45DecisionTree import C45DecisionTree


class TestDecisionTree(TestCase):
    def setUp(self) -> None:
        self.dataset = datasets.load_iris()
        self.all_categorical_feature = False
        self.max_depth = 3
        self.min_sample_leaf = 4
        self.split_criterion = "entropy"
        self.tree_type = "classification"
        # tree_type = "regression"
        X = self.dataset.data
        Y = self.dataset.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, train_size=0.8)

    def test_Sklearn_Decision(self):
        if self.tree_type == "classification":
            model = DecisionTreeClassifier(criterion=self.split_criterion, max_depth=self.max_depth, min_samples_leaf=self.min_sample_leaf)
        else:
            model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_sample_leaf)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        if self.tree_type == "classification":
            print(classification_report(y_true=self.y_test, y_pred=y_pred))
        else:
            print(mean_squared_error(self.y_test, y_pred))

    def test_ID3DecisionTree(self):
        # convert continuous feature to categorical features
        # ID3 can only fit categorical data
        f = lambda x: int(x)
        func = np.vectorize(f)
        X = func(self.dataset.data)
        Y = self.dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
        # model fit
        model = ID3DecisionTree(max_depth=self.max_depth, min_sample_leaf=self.min_sample_leaf, verbose=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if self.tree_type == "classification":
            print(classification_report(y_true=y_test, y_pred=y_pred))
        else:
            print(mean_squared_error(y_test, y_pred))

    def test_C45DecisionTre(self):
        model = C45DecisionTree(max_depth=self.max_depth, min_sample_leaf=self.min_sample_leaf, verbose=True)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        if self.tree_type == "classification":
            print(classification_report(y_true=self.y_test, y_pred=y_pred))
        else:
            print(mean_squared_error(self.y_test, y_pred))

    def test_CART(self):
        model = CART(max_depth=self.max_depth, min_sample_leaf=self.min_sample_leaf, verbose=True,
                     tree_type=self.tree_type,
                     split_criterion=self.split_criterion)
        model.fit(self.X_train, self.y_train)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        if self.tree_type == "classification":
            print(classification_report(y_true=self.y_test, y_pred=y_pred))
        else:
            print(mean_squared_error(self.y_test, y_pred))
