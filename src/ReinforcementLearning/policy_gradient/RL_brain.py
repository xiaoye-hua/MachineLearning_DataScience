#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 7:23 下午
# @Author  : guohua08
# @File    : QLearningRL_brain.py
import tensorflow as tf
import numpy as np
from BaseClass.RLBrainBase import RLBrainBase


class PolicyGradient(RLBrainBase):
    def __init__(self, feature_num, label_num, learning_rate=0.01, reward_decay=0.95):
        super(RLBrainBase, self).__init__()

# class PolicyGradient:
#     def __init__(self, feature_num, label_num, learning_rate=0.01, reward_decay=0.95):
        self.feature_num = feature_num
        self.label_num = label_num
        self.lr = learning_rate
        self.gamma = reward_decay
        self._build_net()
        self.actions = []
        self.rewards = []
        self.observation = []
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("input"):
            self.tf_obs = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_num), name="observation")
            self.tf_acs = tf.placeholder(dtype=tf.int32, shape=(None, ), name="actions")
            self.tf_rs = tf.placeholder(dtype=tf.float32, shape=(None, ), name="return")
        layer = tf.layers.dense(
            inputs=self.tf_obs
            , units=10
            , activation=tf.nn.tanh
            , kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3)
            , bias_initializer=tf.constant_initializer(0.1)
            , name="fc1"
        )
        self.out = tf.layers.dense(
            inputs=layer
            , units=self.label_num
            , activation=None
            , kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3)
            , bias_initializer=tf.constant_initializer(0.1)
            , name="fc2"
        )
        self.out_prob = tf.nn.softmax(logits=self.out)
        with tf.name_scope("loss"):
            log = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.tf_acs)
            # TODO: tf details code  --> FINISHED
            # TODO： 为啥会到这个公式？
            #log = tf.reduce_sum(-tf.log(self.out) * tf.one_hot(self.tf_acs, self.label_num), axis=1)
            #   TODO: 网络的区别在？？
            loss = tf.reduce_mean(log*self.tf_rs)
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

    def take_action(self, observation):
        all_prob = self.session.run(self.out_prob, feed_dict={self.tf_obs:observation[np.newaxis, :]})
        # choose action based on probability distribution
        action = np.random.choice(range(all_prob.shape[1]), p=all_prob.ravel())
        return action

    def store(self, reward, action, observation):
        self.actions.append(action)
        self.rewards.append(reward)
        self.observation.append(observation)

    def learn(self):
        new_rewards = self._discount_and_norm_rewards()
        self.session.run(self.train_op, feed_dict={
            self.tf_acs: np.array(self.actions)
            , self.tf_obs: np.vstack(self.observation)
            , self.tf_rs: new_rewards
        })
        self.actions = []
        self.rewards = []
        self.observation = []

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


