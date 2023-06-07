#!/usr/bin/python
# -*- coding:utf8 -*-

#
# import torch
# from torch import nn
# from ..common import logsigma2cov, split_first_dim, merge_first_two_dims, softplus, DBlock
# from ..common import DiagMultivariateNormal as MultivariateNormal, MLP, _stable_division
# from .ct_common import dt2ts, repeat_tail
# from ..func import normal_differential_sample, multivariate_normal_kl_loss, kld_gauss, zeros_like_with_shape
# from einops import rearrange
# from . import ODEFunc
# from .diffeq_solver import solve_diffeq
# from .interpolation import interpolate
# from .. import BaseModel
# import torch.nn.functional as F
#
# import torch
# from torch import distributions, nn, optim
#
# import torchsde
# sdeint_fn = torchsde.sdeint
#
# from torchdiffeq import odeint as odeint
#
# from ..common import LinearScheduler, EMAMetric
# from ..common import PreProcess, DBlock
#
# class ODEDynamic(nn.Module, BaseModel):
#     def __init__(self, h_size, y_size, u_size, hidden_size):
#         super(ODEDynamic, self).__init__()
#         u_size -= 1  # The last dimension of input variable represents dt
#         self.y_size, self.u_size, self.h_size = y_size, u_size, h_size
#
#         self.process_u = PreProcess(u_size, h_size)
#         self.process_x = PreProcess(y_size, h_size)
#
#         # dy/dt = f(y, u, θ)
#         self.gradient_net = MLP(h_size*2, hidden_size, h_size, num_mlp_layers=1)
#         self.ode_func = ODEFunc(
#             ode_net=self.gradient_net,
#             inputs_interpolation=None,
#             ode_type='normal'
#         )
#     def predictor(self, y0, current_t, predict_t):
#         y = odeint(
#             func=self.ode_func,
#             y0=y0,
#             t=predict_t
#         )
#
#
#     def _forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
#         outputs = {
#             'state_mu': observations_seq,
#             'state_logsigma': -torch.ones_like(observations_seq) * float('inf'),
#         }
#         pred_next_ob = self.predictor(
#             torch.cat([self.process_u(external_input_seq[-1]), self.process_x(observations_seq[-1])], dim=-1)
#         )
#
#         return outputs, {'yn': pred_next_ob}
#
#     def _forward_prediction(self, external_input_seq, n_traj=1, memory_state=None):
#
#         l, batch_size, _ = external_input_seq.size()
#
#         external_input_seq_embed = self.process_u(external_input_seq)
#
#         yn = zeros_like_with_shape(external_input_seq, (batch_size, self.observations_size)
#                                    ) if memory_state is None else memory_state['yn']
#
#         predicted_seq = []
#         for t in range(l):
#             predicted_seq.append(yn)
#             yn = self.predictor(torch.cat([external_input_seq_embed[t], self.process_x(yn)], dim=-1))
#         predicted_seq = torch.stack(predicted_seq, dim=0)
#         predicted_dist = MultivariateNormal(
#             predicted_seq, torch.diag_embed(torch.zeros_like(predicted_seq))
#         )
#
#         predicted_seq_sample = predicted_seq.unsqueeze(dim=-2).repeat(1, 1, n_traj, 1)
#         outputs = {
#             'predicted_seq_sample': predicted_seq_sample,
#             'predicted_dist': predicted_dist,
#             'predicted_seq': predicted_seq
#         }
#         return outputs, {'yn': yn}
#
#     def call_loss(self, external_input_seq, observations_seq, memory_state=None):
#         """
#         Teacher forcing模式训练
#         :param external_input_seq:
#         :param observations_seq:
#         :param memory_state:
#         :return:
#         """
#
#         pred_ob = self.predictor(
#             torch.cat([self.process_u(external_input_seq[:-1]), self.process_x(observations_seq[:-1])], dim=-1)
#         )
#         # outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)
#         # predicted_seq = outputs['x_seq']
#         return {
#             'loss': F.mse_loss(pred_ob, observations_seq[1:]),
#             'kl_loss': 0,
#             'likelihood_loss': 0
#         }
#
#     def decode_observation(self, outputs, mode='sample'):
#         """
#         Args:
#           outputs:
#           mode: dist or sample
#
#         Returns:
#           model为sample时，从分布采样(len,batch_size,observation)
#           为dist时，直接返回分布对象torch.distributions.MultivariateNormal
#
#         方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
#         """
#         if mode == 'dist':
#             observations_normal_dist = MultivariateNormal(
#                 outputs['state_mu'], logsigma2cov(outputs['state_logsigma'])
#             )
#             return observations_normal_dist
#         elif mode == 'sample':
#             return outputs['state_mu']



import matplotlib.pyplot as plt
# from scipy.integrate import odeint
import numpy as np
import torch
from torchdiffeq import odeint
import math
# from torchdiffeq import odeint_adjoint

def diff(t, y):
    # dy/dx = x  => x为t
    # return torch.from_numpy(np.array(x))
    return torch.from_numpy(np.array(t))

t = torch.linspace(0, 10, 11)  # 给出x范围


y = odeint(diff, torch.from_numpy(np.array(10.)), t)  # 设初值为0 此时y为一个数组，元素为不同x对应的y值

plt.plot(t, y)
plt.grid()
plt.show()
""""""

