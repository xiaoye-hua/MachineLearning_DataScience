# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/1 上午11:01
# @Disc    :
import pandas as pd
import numpy as np


def get_col_type_dist(df: pd.DataFrame) -> None:
    """
    get distribution of dataframe column for feature creating
    Args:
        df:
    Returns:
    """
    types = pd.DataFrame(df.dtypes)
    type_dict = {}
    for row in types.iterrows():
        try:
            type_dict[row[1][0]].append(row[0])
        except:
            type_dict[row[1][0]] = [row[0]]
    for key, value in type_dict.items():
        print('*'*20)
        print(f'Column Type: {key}')
        print(f'Column number: {len(type_dict[key])}')
    print('String type columns:')
    print(type_dict[np.dtype('O')])