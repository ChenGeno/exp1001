#!/usr/bin/python
import sys
# -*- coding:utf8 -*-
import numpy as np
import os

import torch.nn as nn
from torch.autograd import Variable
import torch
from Control_Exp1001.control.base_ac import ACBase
from . import HDP
from Control_Exp1001.system_modeling.model_render import SystemModel
from Control_Exp1001.common.normalize import NoNormalizer


class HCNVI(HDP, ACBase):
    def __init__(self,
                 y_size,
                 u_size,
                 c_size,
                 dynamic_model_config_override_str,
                 penalty,
                 normalizer=None,
                 gpu_id=1,
                 replay_buffer=None,
                 u_bounds=None,
                 exploration=None,
                 gamma=0.6,

                 ac_train_batch_size=1,
                 critic_nn_error_limit=1,
                 critic_nn_lr=0.01,
                 hidden_critic=14,
                 Nc=1000,
                 u_optim='sgd',
                 exp_path=None
                 ):

        ACBase.__init__(self, y_size, u_size, c_size, gpu_id=gpu_id, replay_buffer=replay_buffer,
                                  u_bounds=u_bounds, exploration=exploration)

        self.model = SystemModel(input_dim=u_size, output_dim=y_size+c_size, override_str=dynamic_model_config_override_str)
        self.penalty_calculator = penalty
        self.normalizer = normalizer if normalizer else NoNormalizer()

        dim_y, dim_u, dim_c = self.y_size, self.u_size, self.c_size

        self.ac_train_batch_size = ac_train_batch_size

        self.delta_u = self.u_bounds[:, 1] - self.u_bounds[:, 0]
        self.mid_u = np.mean(self.u_bounds, axis=1)
        self.u_optim = u_optim

        self.predict_training_losses = []
        self.critic_nn_error_limit = critic_nn_error_limit

        #定义critic网络相关:HDP
        self.critic_nn = nn.Sequential(
            nn.Linear(dim_y+dim_y+dim_c, hidden_critic, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_critic, 1),
        )
        self.critic_nn_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=critic_nn_lr)
        self.critic_criterion = torch.nn.MSELoss()
        self.gamma = gamma

        self.Nc = Nc
        self.log_y = []
        self.exp_path = exp_path

        self.diff_U = torch.FloatTensor(self.u_bounds[:, 1] - self.u_bounds[:, 0]).to(self.device)
        self.mid_u = torch.FloatTensor(self.mid_u).to(self.device)

        self.indices_y_tensor = torch.LongTensor(self.indice_y).to(self.device)
        self.indices_c_tensor = torch.LongTensor(self.indice_c).to(self.device)
        self.indices_y_star_tensor = torch.LongTensor(self.indice_y_star).to(self.device)
        self.penalty_S = torch.FloatTensor(self.penalty_calculator.S).to(self.device)

    def _act(self, state):

        y_star, y, c = [torch.FloatTensor(self.normalizer.normalize(state[ind], c)).to(self.device).unsqueeze(dim=0)
                         for c, ind in zip(
                'yyc', [self.indice_y_star, self.indice_y, self.indice_c]
            )]
        y_inversed = self.normalizer.inverse(y, 'y')
        y_star_inversed = self.normalizer.inverse(y_star, 'y')
        c_inversed = self.normalizer.inverse(c, 'c')

        try:
            u_bound_min = self.normalizer.normalize(torch.FloatTensor(self.u_bounds[:, 0]), 'u').to(self.device)
            u_bound_max = self.normalizer.normalize(torch.FloatTensor(self.u_bounds[:, 1]), 'u').to(self.device)
        except:
            u_bound_min = torch.FloatTensor(self.u_bounds[:, 0]).to(self.device)
            u_bound_max = torch.FloatTensor(self.u_bounds[:, 1]).to(self.device)

        act = torch.nn.Parameter(torch.rand((1, self.u_size)).to(self.device))
        act.data = torch.min(torch.max(act.data, u_bound_min), u_bound_max)

        if self.u_optim == "adam":
            opt = torch.optim.Adam(params=[act], lr=0.1)
        elif self.u_optim == 'sgd':
            opt = torch.optim.SGD(params=[act], lr=0.8)
        elif self.u_optim == 'RMSprop':
            opt = torch.optim.RMSprop(params=[act], lr=0.01)
        elif self.u_optim == 'adagrad':
            opt = torch.optim.Adagrad(params=[act], lr=0.1)
        else:
            opt = torch.optim.SGD(params=[act], lr=0.8)

        u_iter_times = 0
        while True:
            old_act = act.clone()
            # penalty = self.penalty_calculator.diff_cal_penalty(y_star, y, act, c)
            # torch.clamp(a, b, c)
            # 计算控制量惩罚的计算图
            penalty = self.penalty_calculator.diff_cal_penalty(
                y_star_inversed,
                y_inversed,
                act,
                c_inversed
            )
            # y.requires_grad = True
            yc_pred = self.model.single_step_predict(
                torch.cat([y, c], dim=-1).unsqueeze(dim=0),
                act.unsqueeze(dim=0),
                n_traj=1,
                pred_type='one',
                grad=True
            )
            y_pred, c_pred = yc_pred[..., :self.y_size], yc_pred[..., self.y_size:]
            J_pred = self.critic_nn(torch.cat([y_pred, y_star, c_pred], dim=-1))
            J_loss = (penalty + self.gamma * J_pred).mean()
            opt.zero_grad()
            J_loss.backward()
            opt.step()
            act.data = torch.min(torch.max(act.data, u_bound_min), u_bound_max)
            u_iter_times += 1

            if torch.dist(act, old_act) < 1e-4:
                break
            if u_iter_times > 4000:
                break

        print('step:', self.step, "u_iter_times: ", u_iter_times)
        return self.normalizer.inverse(act[0], 'u').detach().cpu().numpy()

    @staticmethod
    def train_actor(*args, **kwargs):
        return 0
