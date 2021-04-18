#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/4 11:27 下午
# @Author  : guohua08
# @File    : RL_brain.py
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List


class ActorBase(metaclass=ABCMeta):
    def choose_action(self, **kwargs):
        pass

    def learn(self, **kwargs):
        pass


class DiscreteActor(ActorBase):
    def __init__(self, sess, feature_num, action_num, learning_rate=0.001):
        self.sess = sess
        self.feature_num = feature_num
        self.action_num = action_num
        self.lr = learning_rate
        self._build_net()

    def _build_net(self):
        self.s = tf.placeholder(name="s", shape=(None, self.feature_num), dtype=tf.float32)
        self.a = tf.placeholder(name="a", shape=(None, ), dtype=tf.int32)
        self.td_error = tf.placeholder(name="td_error", shape=(None, ), dtype=tf.float32)
        with tf.variable_scope("actor_net"):
            l1 = tf.layers.dense(
                inputs=self.s
                , name="l1"
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
                , activation=tf.nn.relu
                , units=20
            )
            self.action_logit = tf.layers.dense(
                inputs=l1
                , name= "action_logit"
                , units=self.action_num
                , activation=None
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
            )
            self.action_prob = tf.nn.softmax(self.action_logit)
        with tf.variable_scope("loss"):
            log_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.action_logit
                , labels=self.a
            )
            self.loss = tf.reduce_mean(log_loss*self.td_error)
        with tf.variable_scope("train_op"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, a, s, td_error):
        feed_dict = {
            self.a: np.array([a])
            , self.s: s[np.newaxis, :]
            , self.td_error: np.array([td_error])
        }
        self.sess.run(
            [self.loss, self.train_op]
            , feed_dict=feed_dict
        )

    def choose_action(self, s):
        action_prob = self.sess.run(
            self.action_prob
            , feed_dict={
                self.s: s[np.newaxis, :]
            }
        )
        return np.random.choice(np.arange(action_prob.shape[1]), p=action_prob.ravel())


class ContinuousActor(ActorBase):
    def __init__(self, sess: tf.Session, feature_num: np.array, action_bound: List,
                 learning_rate=0.001
                 ) -> None:
        self.feature_num = feature_num
        self.action_bound = action_bound
        self.lr = learning_rate
        self.sess = sess
        self._build_net()

    def _build_net(self) -> None:
        self.s = tf.placeholder(
            shape=(None, self.feature_num)
            , dtype=tf.float32
            , name="observation"
        )
        self.a = tf.placeholder(
            shape=None   # or shape=None
            , dtype=tf.float32
            , name="action"
        )
        self.td_error = tf.placeholder(
            shape=None
            , dtype=tf.float32
            , name="td_error"
        )
        with tf.variable_scope("actor"):
            l1 = tf.layers.dense(
                inputs=self.s
                , name="l1"
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
                , activation=tf.nn.relu
                , units=30
            )
            mu = tf.layers.dense(
                inputs=l1
                , name="mu"
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
                , activation=tf.nn.tanh
                , units=1
            )
            sigma = tf.layers.dense(
                inputs=l1
                , name="sigma"
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
                , activation=tf.nn.softplus
                , units=1
            )
            # TOD0： why add 0.1
            self.mu, self.sigma = tf.squeeze(2*mu), tf.squeeze(sigma + 0.1)
            self.dist = tf.distributions.Normal(self.mu, self.sigma)
            with tf.variable_scope("loss"):
                log_prob = self.dist.log_prob(self.a)
                self.loss = log_prob*self.td_error
                # 增加探索性
                self.loss += self.dist.entropy() * 0.01
                self.loss = -self.loss
            # TODO 没懂？？
            global_step = tf.Variable(0, trainable=False)
            with tf.variable_scope("train"):
                self.train_ops = tf.train.AdamOptimizer(self.lr).minimize(self.loss
                                                                          , global_step
                                                                          )


            # variable for inference
            self.inference_action = tf.clip_by_value(self.dist.sample(1), self.action_bound[0],
                                                     self.action_bound[1])

    def learn(self, a: np.array, s: np.array, td_error: float) -> None:
        feed_dict = {
            self.a: a
            , self.s: s[np.newaxis, :]
            , self.td_error: td_error
        }
        self.sess.run(
            self.train_ops
            , feed_dict
        )

    def choose_action(self, s: np.array) -> List[float]:
        action = self.sess.run(
            self.inference_action
            , {
                self.s: s[np.newaxis, :]
            }
        )
        return action


class Critic:
    """
    online learning for value function V(s)
    """
    def __init__(self, sess, feature_num, learning_rate=0.01, reward_decay=0.9):
        self.sess = sess
        self.feature_num = feature_num
        self.gamma = reward_decay
        self.lr = learning_rate
        self._build_net()

    def _build_net(self):
        self.s = tf.placeholder(
            name="s"
            , shape=(None, self.feature_num)
            , dtype=tf.float32
        )
        self.reward = tf.placeholder(
            name="reward"
            , shape=None
            , dtype=tf.float32
        )
        self.v_next= tf.placeholder(
            name="v_next"
            , shape=None
            , dtype=tf.float32
        )
        with tf.variable_scope("value_network"):
            l1 = tf.layers.dense(
                inputs=self.s
                , name="l1"
                , units=20
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
                , activation=tf.nn.relu
            )
            self.exp_v = tf.layers.dense(
                inputs=l1
                , name="exp_v"
                , units=1
                , kernel_initializer=tf.random_normal_initializer(0., 0.1)
                , bias_initializer=tf.constant_initializer(0.1)
                # , activation=tf.nn.relu
            )
        with tf.variable_scope("td_error"):
            self.td_error = self.reward + self.gamma * self.v_next - self.exp_v
            loss = tf.square(self.td_error)
        with tf.variable_scope("train_op"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def learn(self, current_state, reward, next_state):
        v_next = self.sess.run(
            self.exp_v
            , feed_dict={
                self.s: next_state[np.newaxis, :]
            }
        )
        _, td_error = self.sess.run(
            [self.train_op, self.td_error]
            , feed_dict={
                self.s: current_state[np.newaxis, :]
                , self.reward: reward
                , self.v_next: v_next
            }
        )
        return td_error[0][0]
