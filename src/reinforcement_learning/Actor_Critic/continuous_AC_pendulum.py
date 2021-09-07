#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/18 5:53 下午
# @Author  : guohua08
# @File    : continuous_AC_pendulum.py
from Actor_Critic.RL_brain import ContinuousActor, Critic
import gym
import numpy as np
import pandas as pd
import tensorflow as tf

# Config
env_name = "Pendulum-v0"
total_episodes = 3000
render = False
render_reward_threshold = -100

env = gym.make(env_name)
env.seed(1)
np.random.seed(2)
tf.random.set_random_seed(2)
action_bound = env.action_space.high
# env.unwrapped

actor_lr = 0.001
critic_lr = 0.01

sess = tf.Session()
actor = ContinuousActor(
    sess=sess
    , feature_num=env.observation_space.shape[0]
    , action_bound=[-action_bound, action_bound]
    , learning_rate=actor_lr
)
critic = Critic(
    sess=sess
    , feature_num=env.observation_space.shape[0]
    , learning_rate=critic_lr
)

sess.run(tf.global_variables_initializer())

current_reward = 0
for episode in range(total_episodes):
    observation = env.reset()
    reward_lst = []
    total_step = 0
    while True:
        total_step += 1
        # if render:
        #     env.render()
        action = actor.choose_action(s=observation)
        observation_, reward, done, info = env.step(action=action)
        # 方便收敛？
        # print(reward)
        reward /= 10
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
                current_reward = 0.9*current_reward + 0.1*sum(reward_lst)
            if current_reward > render_reward_threshold:
                render = True
            print(f"Episode num: {episode}; running rewards: {int(current_reward)}; total step in this episode: {total_step}")
            break


