# -*- coding: utf-8 -*-
# @File    : BaseDataConvertor.py
# @Author  : Hua Guo
# @Disc    :
import os
from abc import ABCMeta, abstractmethod


class BaseDataConvertor(metaclass=ABCMeta):

    @abstractmethod
    def convert(self):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)