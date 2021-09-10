# -*- coding: utf-8 -*-
# @File    : CART.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午9:04
# @Disc    :
from typing import List

import numpy as np

from src.Model.DecisionTree import DecisionTree, TreeNode


class CART(DecisionTree):

    def __init__(self, max_depth, min_sample_leaf, split_criterion="gini", tree_type="classification", min_split_criterion=1e-4, verbose=False):
        super(CART, self).__init__(max_depth=max_depth, min_sample_leaf=min_sample_leaf, min_split_criterion=min_split_criterion
                                   , verbose=verbose)
        self.tree_type = tree_type
        self.split_criterion = split_criterion
        assert self.split_criterion in ["gini", "entropy"]
        assert self.tree_type in ["classification", "regression"]

    def split_node(self, node: TreeNode) -> List[TreeNode]:
        child_node_lst = []
        child_cate_order = None
        gini_index = float("inf")
        split_col = None
        for col_idx in range(self.feature_num):
            current_child_cate_order = list(np.unique(self.data[node.data_idx][:, col_idx]))
            current_child_cate_order.sort()
            for col_value in current_child_cate_order:
                left_data_idx = np.intersect1d(node.data_idx, np.where(self.data[:, col_idx] <= col_value))
                right_data_idx = np.intersect1d(node.data_idx, np.where(self.data[:, col_idx] > col_value))
                current_child_node_lst = []
                if len(left_data_idx) != 0:
                    left_tree = TreeNode(
                            data_idx=left_data_idx,
                            depth=node.depth+1,
                        )
                    current_child_node_lst.append(left_tree)
                if len(right_data_idx) != 0:
                    right_tree = TreeNode(
                            data_idx=right_data_idx,
                            depth=node.depth+1,
                        )
                    current_child_node_lst.append(right_tree)
                current_gini_index = self.get_split_criterion(node, current_child_node_lst)
                if current_gini_index < gini_index:
                    gini_index = current_gini_index
                    child_node_lst = current_child_node_lst
                    child_cate_order = col_value
                    split_col = col_idx
        node.child = child_node_lst
        node.set_attribute(split_col=split_col, child_cate_order=child_cate_order)
        return child_node_lst

    def get_split_criterion(self, node, child_node_lst):
        total = len(node.data_idx)
        split_criterion = 0
        for child_node in child_node_lst:
            impurity = self.get_impurity(child_node.data_idx)
            split_criterion += len(child_node.data_idx) / float(total) * impurity
        return split_criterion

    def get_impurity(self, data_ids):
        target_y = self.labels[data_ids]
        total = len(target_y)
        if self.tree_type == "regression":
            res = 0
            mean_y = np.mean(target_y)
            for y in target_y:
                res += (y - mean_y) ** 2 / total
        elif self.tree_type == "classification":
            if self.split_criterion == "gini":
                res = 1
                unique_y = np.unique(target_y)
                for y in unique_y:
                    num = len(np.where(target_y==y)[0])
                    res -= (num/float(total))**2
            elif self.split_criterion == "entropy":
                unique, count = np.unique(target_y, return_counts=True)
                res = 0
                for c in count:
                    p = float(c) / total
                    res -= p * np.log(p)
        return res

    def get_nex_node(self, node: TreeNode, x: np.array):
        col_value = x[node.split_col]
        if col_value> node.child_cate_order:
            index = 1
        else:
            index = 0
        return node.child[index]