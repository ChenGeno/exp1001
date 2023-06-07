#!/usr/bin/python
# -*- coding:utf8 -*-
import sys
from abc import ABC

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


class DHP(HDP):
    def __init__(self, **para):
        super(DHP, self).__init__(**para)
        hidden_critic = para['hidden_critic']
        critic_nn_lr = para['critic_nn_lr']
        self.critic_nn = nn.Sequential(
            nn.Linear(self.y_size * 2 + self.c_size, hidden_critic, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_critic, self.y_size, bias=False),
        )
        self.critic_nn_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=critic_nn_lr)
        self.critic_criterion = torch.nn.MSELoss()

    def critic_loss(self, y, y_star, c, ny, ny_star, nc, penalty):
        batch_size, _ = y.shape
        y_tmp = y.clone()
        y_tmp.requires_grad = True
        y_tmp.grad = None
        action = self.actor_nn(torch.cat([y_tmp, y_star, c], dim=1))
        penalty = self.penalty_calculator.diff_cal_penalty(
            self.normalizer.inverse(y_star, 'y'),
            self.normalizer.inverse(y_tmp, 'y'),
            self.normalizer.inverse(action, 'u'),
            self.normalizer.inverse(c, 'c'),
        )
        penalty.backward(torch.ones_like(penalty)/batch_size, retain_graph=True)
        yc_pred = self.model.single_step_predict(
            torch.cat([y, c], dim=-1).unsqueeze(dim=0),
            action.unsqueeze(dim=0),
            n_traj=1,
            pred_type='one',
            grad=True
        )
        y_pred, c_pred = yc_pred[..., :self.y_size], yc_pred[..., self.y_size:]
        y_pred_co_state = self.critic_nn(torch.cat((y_pred, y_star, nc), dim=1))

        y_pred.backward(self.gamma * y_pred_co_state / batch_size)

        y_co_state = self.critic_nn(torch.cat([y_tmp, y_star, c], dim=1))

        critic_loss = self.critic_criterion(y_co_state, Variable(y_tmp.grad.data))
        return critic_loss

    def actor_loss(self, y, y_star, c, ny, n_y_star, nc):
        action = self.actor_nn(torch.cat([y, y_star, c], dim=1))

        # 计算控制量惩罚的计算图
        penalty = self.penalty_calculator.diff_cal_penalty(
            self.normalizer.inverse(y_star, 'y'),
            self.normalizer.inverse(y, 'y'),
            self.normalizer.inverse(action, 'u'),
            self.normalizer.inverse(c, 'c'),
        )

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

        # J(k+1) = U(k)+J(y(k+1),c)

        y_pred_co_state = self.critic_nn(torch.cat((y_pred, y_star, c_pred), dim=-1)).detach()
        J_loss = penalty + self.gamma * (y_pred.mul(y_pred_co_state).sum(dim=-1))
        return J_loss.mean()

