# -*- coding: utf-8 -*-
# @File    : test_SVM.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午9:54
# @Disc    :
import numpy as np
from unittest import TestCase
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.svm import SVC

from src.Model.SVM import LinearSVM


class TestSVM(TestCase):
    def setUp(self) -> None:
        self.dataset = datasets.load_iris()
        self.tree_type = "classification"
        X = self.dataset.data
        Y = self.dataset.target
        # map the target from 3 classes to 2 classed
        for idx in range(len(Y)):
            if Y[idx] == 2:
                Y[idx] = np.random.choice([0, 1])
                Y[idx] = float(Y[idx])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, train_size=0.8)

    def test_Sklearn_Decision(self):
        print(self.X_train.shape)
        model = SVC(gamma='auto', C=1.)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print(classification_report(y_true=self.y_test, y_pred=y_pred))

    def test_LinearSVM_Binary_Hard(self):
        model = LinearSVM()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print(classification_report(y_true=self.y_test, y_pred=y_pred))

    def test_LinearSVM_Binary_Soft(self):
        model = LinearSVM()
        model.fit(self.X_train, self.y_train, soft=True)
        y_pred = model.predict(self.X_test)
        print(classification_report(y_true=self.y_test, y_pred=y_pred))