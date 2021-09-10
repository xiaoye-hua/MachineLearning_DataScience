# -*- coding: utf-8 -*-
# @File    : ID3DecisionTree.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午9:03
# @Disc    :
import numpy as np

from src.Model.DecisionTree import DecisionTree, TreeNode


class ID3DecisionTree(DecisionTree):

    def split_node(self, node):
        child_node_lst = []
        child_cate_order = []
        informatin_gain = 0
        split_col = None
        for col_idx in range(self.feature_num):
            current_child_cate_order = list(np.unique(self.data[node.data_idx][:, col_idx]))
            current_child_node_lst = []
            for col_value in current_child_cate_order:
                data_idx = np.intersect1d(node.data_idx, np.where(self.data[:, col_idx] == col_value))
                current_child_node_lst.append(
                    TreeNode(
                        data_idx=data_idx,
                        depth=node.depth+1
                    )
                )
            current_gain = self.get_split_criterion(node, current_child_node_lst)
            if current_gain > informatin_gain:
                informatin_gain = current_gain
                child_node_lst = current_child_node_lst
                child_cate_order = current_child_cate_order
                split_col = col_idx
        if informatin_gain<self.min_split_criterion:
            return
        else:
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
        target_Y = self.labels[data_ids]
        total = len(target_Y)
        unique, count = np.unique(target_Y, return_counts=True)
        res = 0
        for c in count:
            p = float(c)/total
            res -= p*np.log(p)
        return res

    def get_nex_node(self, node, x):
        try:
            next_node = node.child[node.child_cate_order.index(x[node.split_col])]
        except:
            next_node = node.child[0]
        return next_node