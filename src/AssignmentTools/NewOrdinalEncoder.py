# -*- coding: utf-8 -*-
# @File    : NewOrdinalEncoder.py
# @Author  : Hua Guo
# @Time    : 2021/10/2 下午10:44
# @Disc    :
from sklearn.preprocessing import OrdinalEncoder
from typing import Optional, List
import pandas as pd


class NewOrdinalEncoder(OrdinalEncoder):
    """
    comparable with null value & numerical input
    """
    def __init__(self, category_cols: List[str]) -> None:
        super(OrdinalEncoder, self).__init__()
        self.ordinal_encoder = OrdinalEncoder()
        self.category_cols = category_cols
        self.null_map = {col: 'null' for col in self.category_cols}

    def fit(self, X, y=None):
        X.fillna(self.null_map, inplace=True)
        self.ordinal_encoder.fit(X[self.category_cols])
        return self

    def transform(self, X):
        X.fillna(self.null_map, inplace=True)
        X[self.category_cols] = self.ordinal_encoder.transform(X[self.category_cols])
        return X


