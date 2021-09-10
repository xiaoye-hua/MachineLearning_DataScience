# -*- coding: utf-8 -*-
# @File    : C45DecisionTree.py
# @Author  : Hua Guo
# @Time    : 2021/9/10 下午9:04
# @Disc    :
import numpy as np

from src.Model.DecisionTree.ID3DecisionTree import ID3DecisionTree


class C45DecisionTree(ID3DecisionTree):

    def get_split_criterion(self, node, child_node_lst):
        total = len(node.data_idx)
        split_criterion = 0
        for child_node in child_node_lst:
            impurity = self.get_impurity(child_node.data_idx)
            split_criterion += len(child_node.data_idx) / float(total) * impurity
        intrinsic_value = self._get_intrinsic_value(node, child_node_lst)
        split_criterion= split_criterion/intrinsic_value
        return split_criterion

    def _get_intrinsic_value(self, node, child_node_lst):
        total = len(node.data_idx)
        res = 0
        for n in child_node_lst:
            frac = len(n.data_idx) / float(total)
            res -=  frac * np.log(frac)
        return res