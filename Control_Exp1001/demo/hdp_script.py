#!/usr/bin/python
# -*- coding:utf8 -*-
import sys
import os
import numpy as np
import torch
import random
import math
import json
from Control_Exp1001.simulation import Thickener
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.common.normalize import EnvNormalizer
from Control_Exp1001.control.adp.hdp import HDP
from Control_Exp1001.common.penaltys.quadratic import Quadratic

from Control_Exp1001.exp import OneRoundExp
from Control_Exp1001.eval import OneRoundEvaluation
from Control_Exp1001.utils import sample_from_env, get_exp_path


EXP_NAME = 'hdp single'
rounds = 1000
seed = random.randint(0, 1000000)
name = 'hdp single'
gpu_id = 0

exp_path = get_exp_path(EXP_NAME)


env = Thickener(
    noise_p=0.03,
    noise_in=True
)
penalty_para = {
    "y_size": env.y_size,
    "u_size": env.u_size,
    "weight_matrix": [0, 0.004],
    "S": [0.0001, 0.0008],
    'u_bounds': env.u_bounds
}
penalty_calculator = Quadratic(**penalty_para)

env.penalty_calculator = penalty_calculator

exploration = No_Exploration()

argv_str = f"use_cuda={'True' if gpu_id >= 0 else 'False'} dataset.dataset_window=1 " \
           f"dataset.history_length=1 dataset.forward_length=15 " \
           f"model=nn train.batch_size=1024 train.epochs=50 train.eval_epochs=5 train.optim.lr=5e-3"

hdp = HDP(
    y_size=env.size_yudc[0],
    u_size=env.size_yudc[1],
    c_size=env.size_yudc[3],
    dynamic_model_config_override_str=argv_str,
    penalty=penalty_calculator,
    gpu_id=gpu_id,
    replay_buffer=ReplayBuffer(capacity=5),
    u_bounds=env.u_bounds,
    # exploration = None,
    exploration=exploration,
    gamma=0.8,
    ac_train_batch_size=5,
    critic_nn_error_limit=0.001,
    actor_nn_error_limit=0.001,
    # 0.005
    actor_nn_lr=0.003,
    critic_nn_lr=0.01,
    hidden_critic=14,
    hidden_actor=14,
    Na=220,
    Nc=1000,
    exp_path=exp_path
)

if hdp.device != torch.device('cpu'):
    hdp.to(hdp.device)

env.reset()

# 采样训练预测model的初始数据
initial_data_collection_rounds = 300

sampling_policy = lambda _: np.random.uniform(hdp.u_bounds[:, 0], hdp.u_bounds[:, 1])

train_initial_dataset = sample_from_env(env, sampling_policy, 3000,
                                        use_cache=True, cache_name_addition='train')
val_initial_dataset = sample_from_env(env, sampling_policy, 500,
                                      use_cache=True, cache_name_addition='val')

u_mean, u_std = train_initial_dataset.x_mean, train_initial_dataset.x_std
y_mean, y_std = train_initial_dataset.y_mean[:env.size_yudc[0]], train_initial_dataset.y_std[:env.size_yudc[0]]
c_mean, c_std = train_initial_dataset.y_mean[env.size_yudc[0]:], train_initial_dataset.y_std[env.size_yudc[0]:]

hdp.normalizer = EnvNormalizer(u_mean, u_std, y_mean, y_std, c_mean, c_std)
hdp.train_identification_model(train_initial_dataset, val_initial_dataset)

thickener_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    "noise_type": 2,
    'time_length': 20,# 浓密机每次仿真20秒
    'random_seed': seed
}

env_hdp = Thickener(
    penalty_calculator=penalty_calculator,
    **thickener_para,
)

res_list = [OneRoundExp(controller=hdp, env=env_hdp, max_step=rounds, exp_name=name).run()]

eval_res = OneRoundEvaluation(res_list=res_list, save_root=exp_path)

eval_res.plot_all(show=False)

# write str in file
with open(os.path.join(exp_path, 'metrics.txt'), 'w') as f:
    eval_result = eval_res.evaluate()
    f.write(str(list(eval_result)))
