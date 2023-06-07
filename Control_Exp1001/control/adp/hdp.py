#!/usr/bin/python
import sys
# -*- coding:utf8 -*-
import numpy as np
import os

import torch.nn as nn
from torch.autograd import Variable
import torch
from Control_Exp1001.control.base_ac import ACBase
from Control_Exp1001.system_modeling.model_render import SystemModel
from Control_Exp1001.common.normalize import NoNormalizer


class HDP(ACBase):
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
                 actor_nn_error_limit=0.1,

                 actor_nn_lr=0.01,
                 critic_nn_lr=0.01,

                 hidden_critic=14,
                 hidden_actor=10,
                 Na=100,
                 Nc=1000,
                 exp_path=None
                 ):

        super(HDP, self).__init__(y_size, u_size, c_size, gpu_id=gpu_id,replay_buffer=replay_buffer,
                                  u_bounds=u_bounds, exploration=exploration)
        self.model = SystemModel(input_dim=u_size, output_dim=y_size+c_size, override_str=dynamic_model_config_override_str)
        self.penalty_calculator = penalty
        self.normalizer = normalizer if normalizer else NoNormalizer()

        dim_y, dim_u, dim_c = self.y_size, self.u_size, self.c_size

        self.ac_train_batch_size = ac_train_batch_size

        self.delta_u = self.u_bounds[:, 1] - self.u_bounds[:, 0]
        self.mid_u = np.mean(self.u_bounds, axis=1)

        self.predict_training_losses = []
        self.critic_nn_error_limit = critic_nn_error_limit
        self.actor_nn_error_limit = actor_nn_error_limit

        #定义actor网络相关
        self.actor_nn = nn.Sequential(
            nn.Linear(2*dim_y+dim_c, hidden_actor, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_actor, dim_u),
        )

        self.actor_nn_optim = torch.optim.Adam(self.actor_nn.parameters(), lr=actor_nn_lr)

        #定义critic网络相关:HDP
        self.critic_nn = nn.Sequential(
            nn.Linear(dim_y+dim_y+dim_c, hidden_critic, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_critic, 1),
        )
        self.critic_nn_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=critic_nn_lr)
        self.critic_criterion = torch.nn.MSELoss()
        self.gamma = gamma

        self.Na = Na
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

        y = self.normalizer.normalize(state[self.indice_y], 'y')
        y_star = self.normalizer.normalize(state[self.indice_y_star], 'y')
        c = self.normalizer.normalize(state[self.indice_c], 'c')

        x = torch.FloatTensor(np.hstack((y, y_star, c))).to(self.device).unsqueeze(0)

        act = self.actor_nn(x).detach().cpu().squeeze(0).numpy()
        act = self.normalizer.inverse(act, 'u')
        # bounds act in u_bounds
        act = np.clip(act, self.u_bounds[:, 0], self.u_bounds[:, 1])
        self.log_y.append(np.copy(y))

        return act

    def _train(self, s, u, ns, r, done):

        # 先放回放池
        self.replay_buffer.push(s, u, r, ns, done)
        # if len(self.replay_buffer) < self.batch_size:
        #     return
        # 从回放池取数据，默认1条
        state, action, penalty, next_state, done = self.replay_buffer.sample(
            # 尽快开始训练，而不能等batchsize满了再开始
            min(len(self.replay_buffer), self.ac_train_batch_size)
        )

        # 更新模型
        self.update_model(state, action, penalty, next_state, done)

    def update_model(self, state, action, penalty, next_state, done):

        # ndarray to tensor
        state = torch.FloatTensor(self.normalizer.normalize(state, 'yyuc')).to(self.device)
        next_state = torch.FloatTensor(self.normalizer.normalize(next_state, 'yyuc')).to(self.device)
        # action = torch.FloatTensor(self.normalizer.normalize(action, 'u')).to(self.device)
        penalty = torch.FloatTensor(penalty).unsqueeze(1).to(self.device)

        # 提取state中的特性数据项
        y = torch.index_select(state, 1, self.indices_y_tensor)
        ny = torch.index_select(next_state, 1, self.indices_y_tensor)
        y_star = torch.index_select(state, 1, self.indices_y_star_tensor)
        ny_star = torch.index_select(next_state, 1, self.indices_y_star_tensor)

        c = torch.index_select(state, 1, self.indices_c_tensor)
        nc = torch.index_select(next_state, 1, self.indices_c_tensor)

        # 循环更新actor网络和critic网路
        loop_time = 0
        # region update critic nn
        loop_time = self.train_critic(y, y_star, c, ny, ny_star, nc, penalty)
        print('step:', self.step, 'critic loop', loop_time)

        loop_time = self.train_actor(y, y_star, c, ny, ny_star, nc, penalty)
        print('step:', self.step, 'actor loop',loop_time)

    def train_critic(self, y, y_star, c, ny, ny_star, nc, penalty):
        loop_time = 0
        while True:
            critic_loss = self.critic_loss(y, y_star, c, ny, ny_star, nc, penalty)
            loop_time += 1
            # 定义TD loss
            self.critic_nn_optim.zero_grad()
            critic_loss.backward()
            self.critic_nn_optim.step()
            if loop_time >= self.Nc:
                break

            if critic_loss < self.critic_nn_error_limit:
                break
        return loop_time
    def train_actor(self, y, y_star, c, ny, ny_star, nc, penalty):
        loop_time = 0
        last_J = np.inf
        while True:
            J_loss = self.actor_loss(y, y_star, c, ny, ny_star, nc)
            self.actor_nn_optim.zero_grad()
            J_loss.backward()
            self.actor_nn_optim.step()

            loop_time += 1
            if abs(J_loss - last_J) < self.actor_nn_error_limit:
                break
            last_J = float(J_loss)
            if J_loss < 1e-4:
                break
            if loop_time > self.Na:
                break
        return loop_time

    def critic_loss(self, y, y_star, c, ny, ny_star, nc, penalty):
        q_value = self.critic_nn(torch.cat((y, y_star, c), dim=-1))
        next_q_value = self.critic_nn(torch.cat((ny, ny_star, nc), dim=-1))

        target_q = penalty + self.gamma * next_q_value
        critic_loss = self.critic_criterion(q_value, Variable(target_q.data))
        return critic_loss

    def actor_loss(self, y, y_star, c, ny, n_y_star, nc):
        action = self.actor_nn(torch.cat([y, y_star, c], dim=1))
        yc_pred = self.model.single_step_predict(
            torch.cat([y, c], dim=-1).unsqueeze(dim=0),
            action.unsqueeze(dim=0),
            n_traj=1,
            pred_type='one',
            grad=True
        )
        # split yc_pred to y and c
        y_pred, c_pred = yc_pred[..., :self.y_size], yc_pred[..., self.y_size:]

        # y_pred.register_hook(lambda grad: print('system modeling grad: ', grad))

        # 计算控制量惩罚的计算图
        penalty = self.penalty_calculator.diff_cal_penalty(
            self.normalizer.inverse(y_star, 'y'),
            self.normalizer.inverse(y, 'y'),
            self.normalizer.inverse(action, 'u'),
            self.normalizer.inverse(c, 'c'),
        )
        # J(k+1) = U(k)+J(y(k+1),c)
        J_pred = self.critic_nn(torch.cat((y_pred, n_y_star, c_pred), dim=1))
        # penalty_u = torch.zeros(J_pred.shape)
        J_loss = penalty + self.gamma * J_pred
        J_loss = J_loss.mean()
        return J_loss

    def train_identification_model(self, train_dataset, val_dataset):
        self.model.start_train(train_dataset, val_dataset, ckpt_path=os.path.join(self.exp_path, self.__class__.__name__, 'system_model'))
