#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


from Control_Exp1001.demo.thickener.value_iterate_sample import ViSample
from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.penaltys.demo_penalty import DemoPenalty
import matplotlib.pyplot as plt
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.common.action_noise.gaussian_noise import GaussianExploration
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer

def new_vi_sample(capacity=2,predict_round=3000):
    replay_vi_sample = ReplayBuffer(capacity=capacity)
    env_VI_sample = Thickener(
        noise_p=0.03,
        noise_in=True,
    )
    exploration = No_Exploration()

    print('make new vi_sample controller')
    vi_sample = ViSample(
        replay_buffer = replay_vi_sample,
        u_bounds = env_VI_sample.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_VI_sample,
        predict_training_rounds=predict_round,
        gamma=0.4,

        batch_size = capacity,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.001,
        actor_nn_error_limit = 0.001,

        actor_nn_lr = 0.005,
        critic_nn_lr = 0.01,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 14,
        hidden_actor = 14,
        predict_epoch= 30,
    )
    env_VI_sample.reset()
    vi_sample.train_identification_model()
    vi_sample.test_predict_model(test_rounds=100)
    return vi_sample
