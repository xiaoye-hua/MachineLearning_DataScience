# -*- coding: utf-8 -*-
# @File    : DecisionTree.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午8:59
# @Disc    :
from scipy import stats
from abc import ABCMeta


class TreeNode:
    def __init__(self, data_idx, depth, child_lst=[]):
        self.data_idx = data_idx
        self.depth = depth
        self.child = child_lst
        self.label = None
        self.split_col = None
        self.child_cate_order = None

    def set_attribute(self, split_col, child_cate_order=None):
        self.split_col = split_col
        self.child_cate_order = child_cate_order

    def set_label(self, label):
        self.label = label


class DecisionTree(metaclass=ABCMeta):
    def __init__(self, max_depth, min_sample_leaf, min_split_criterion=1e-4, verbose=False):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.verbose = verbose
        self.min_split_criterion = min_split_criterion
        self.root = None
        self.data = None
        self.labels = None
        self.feature_num = None

    def fit(self, X, y):
        """
        X: train data, dimensition [num_sample, num_feature]
        y: label, dimension [num_sample, ]
        """
        self.data = X
        self.labels = y
        num_sample, num_feature = X.shape
        self.feature_num = num_feature
        data_idx = list(range(num_sample))
        self.root = TreeNode(data_idx=data_idx, depth=0, child_lst=[])
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.depth>self.max_depth or len(node.data_idx)==1:
                self.set_label(node)
            else:
                child_nodes = self.split_node(node)
                if not child_nodes:
                    self.set_label(node)
                else:
                    queue.extend(child_nodes)

    def predict(self, X):
        num_sample, num_feature = X.shape
        labels = []
        for idx in range(num_sample):
            x = X[idx]
            node = self.root
            while node.child:
                node = self.get_nex_node(node, x)
            labels.append(node.label)
        return labels

    @classmethod
    def get_split_criterion(self, node, child_node_lst):
        pass

    def set_label(self, node):
        target_Y = self.labels[node.data_idx]
        target_label = stats.mode(target_Y).mode[0]
        node.set_label(label=target_label)

    @classmethod
    def split_node(self, node):
        pass

    @classmethod
    def get_nex_node(self, node, x):
        pass
