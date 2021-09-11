# -*- coding: utf-8 -*-
# @File    : AbstractClass.py
# @Author  : Hua Guo
# @Time    : 2020/2/3 下午6:29
# @Disc    :
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    """
    Abstract class for machine learning model
    """
    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass
