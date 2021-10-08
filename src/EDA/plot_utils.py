# -*- coding: utf-8 -*-
# @File    : plot_utils.py
# @Author  : Hua Guo
# @Time    : 2021/10/3 上午10:39
# @Disc    :
import matplotlib.pyplot as plt
import seaborn as sns


def category_view_binary(df, col, target):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplot(121)
    sns.countplot(data=df, x=col,
                  # order of x
                  order=sorted(df[col].dropna().unique()))
    plt.subplot(122)
    sns.countplot(data=df, x=col
                  , hue=target
                  # order of x
                  , order=sorted(df[col].dropna().unique()))
    plt.show()