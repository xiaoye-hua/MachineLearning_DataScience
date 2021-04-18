#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/4 11:26 下午
# @Author  : guohua08
# @File    : discrete_AC_cartpole.py
from Actor_Critic.RL_brain import DiscreteActor, Critic
import gym
import numpy as np
import pandas as pd
import tensorflow as tf

# Config
env_name = "CartPole-v0"
total_episodes = 3000

env = gym.make(env_name)
env.seed(1)
np.random.seed(2)
tf.random.set_random_seed(2)
# env.unwrapped


sess = tf.Session()
actor = DiscreteActor(
    sess=sess
    , feature_num=env.observation_space.shape[0]
    , action_num=env.action_space.n
)
critic = Critic(
    sess=sess
    , feature_num=env.observation_space.shape[0]
)
sess.run(tf.global_variables_initializer())

current_reward = 0
for episode in range(total_episodes):
    observation = env.reset()
    reward_lst = []
    total_step = 0
    while True:
        total_step += 1
        # env.render()
        action = actor.choose_action(s=observation)
        observation_, reward, done, info = env.step(action=action)
        # 回合结束的惩罚，第一次往写了：
        #      cartpole-v0 返回的reward都是1 （https://github.com/openai/gym/issues/1682）
        if done:
            reward = -20
        td_error = critic.learn(
            current_state=observation
            , reward=reward
            , next_state=observation_
        )
        actor.learn(
            a=action
            , s=observation
            , td_error=td_error
        )
        observation = observation_
        reward_lst.append(reward)
        if done:
            if episode == 0:
                current_reward = sum(reward_lst)
            else:
                current_reward = 0.95*current_reward + 0.05*sum(reward_lst)
            print(f"Episode num: {episode}; running rewards: {int(current_reward)}; total step in this episode: {total_step}")
            break


