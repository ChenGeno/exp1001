#!/usr/bin/python
# -*- coding:utf8 -*-


import torch
from torch import nn
from ..common import logsigma2cov, split_first_dim, merge_first_two_dims, softplus, DBlock
from ..common import DiagMultivariateNormal as MultivariateNormal, MLP, _stable_division, PreProcess, NCDEFunc
from .ct_common import dt2ts, repeat_tail
from ..func import normal_differential_sample, multivariate_normal_kl_loss, kld_gauss, zeros_like_with_shape
from einops import rearrange
from . import ODEFunc
from .diffeq_solver import solve_diffeq
from .interpolation import interpolate
from .. import BaseModel
import math
import torch
from torch import distributions, nn, optim
import torchcde
import torch.nn.functional as F


import torchsde
sdeint_fn = torchsde.sdeint

from torchdiffeq import odeint as odeint

from ..common import LinearScheduler, EMAMetric


class GunnarODE(nn.Module, BaseModel):

    def __init__(self, h_size, y_size, u_size, theta=1.0, mu=0.0, sigma=0.5, inter='gp',
                 dt=1e-2, rtol=1e-3, atol=1e-3, method='euler', adaptive=False, interpolation="cubic"):
        super(GunnarODE, self).__init__()

        u_size -= 1 # The last dimension of input variable represents dt
        self.y_size, self.u_size, self.h_size, self.inter = y_size, u_size, h_size, inter
        self.dt, self.rtol, self.atol, self.method, self.adaptive = dt, rtol, atol, method, adaptive

        # self.process_u = PreProcess(u_size, h_size)
        # self.process_x = PreProcess(y_size, h_size)

        # TODO：batchsize
        self.cde_func = NCDEFunc(h_size, u_size+1, batch_size=512)
        self.interpolation = interpolation
        self.encoder = torch.nn.Linear(u_size+1, h_size)
        # self.decoder = torch.nn.Linear(h_size, y_size)


        # RNN相关
        self.process_u = PreProcess(self.u_size, h_size)
        self.process_x = PreProcess(y_size, h_size)
        self.Ly_gauss = DBlock(2 * h_size, 2 * h_size, y_size)
        self.rnn_encoder = torch.nn.GRU(2 * h_size, h_size, 1)
        self.decoder = MLP(h_size, 2 * h_size, y_size, 1)
        # self.rnn_cell = nn.GRUCell(
        #     h_size*2,
        #     h_size*2
        # )

    def forward(self, ts, us, h0):

        # 预处理，维度适应, 从(l, batch, v) -> (batch, l, v)
        ts = ts.permute(1, 0, 2)
        us = us.permute(1, 0, 2)


        x = torch.cat([ts, us], dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)

        # # TODO: 定义y0（z0）
        # # z0 = ys[:, 0, :]
        # z0 = torch.zeros(batch_size, self.u_size+1).to(us)
        # z0 = self.encoder(z0)

        ys_predict = torchcde.cdeint(X=X,
                             func=self.cde_func,
                             z0=h0,
                             t=ts.squeeze(-1)[0, :])  # 每个batch的t序列是相等的，取第0维

        ys_predict = self.decoder(ys_predict)

        ys_predict = ys_predict.permute(1, 0, 2)

        return ys_predict


    def _forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        l, batch_size, _ = external_input_seq.size()
        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        hn = torch.zeros((batch_size, self.h_size),
                         device=external_input_seq.device) if memory_state is None else memory_state['hn']

        h_seq = [hn]
        x_seq = []
        for t in range(l):
            # encoder网络更新h_t: u_t+1, x_t+1, h_t -> h_t+1 u_t, x_t, h_t -> h_{t+1}
            output, _ = self.rnn_encoder(
                torch.cat([external_input_seq_embed[t], observations_seq_embed[t]], dim=-1).unsqueeze(dim=0),
                hn.unsqueeze(dim=0)
            )
            hn = output[0]
            # x重构 for loss
            x_t = self.decoder(hn)

            h_seq.append(hn)
            x_seq.append(x_t)

        h_seq = torch.stack(h_seq, dim=0)
        h_seq = h_seq[:-1]
        x_seq = torch.stack(x_seq, dim=0)

        outputs = {
            'state_mu': x_seq,
            'state_logsigma': -torch.ones_like(x_seq) * float('inf'),
            'h_seq': h_seq,
            'x_seq': x_seq,
            'observations_seq_embed': observations_seq_embed,
        }

        return outputs, {'hn': hn}


    def _forward_prediction(self, external_input_seq, n_traj=1, memory_state=None):
        l, batch_size, _ = external_input_seq.size()
        input_n_traj = n_traj
        n_traj = 1

        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        ts = dt2ts(dt)

        hn = zeros_like_with_shape(external_input_seq, (batch_size, self.h_size)
                                   ) if memory_state is None else memory_state['hn']


        ys = self.forward(ts, repeat_tail(external_input_seq), hn)
        ys, y = ys[:-1], ys[-1, :batch_size]
        predicted_seq = ys
        predicted_dist = MultivariateNormal(
            predicted_seq, torch.diag_embed(torch.zeros_like(predicted_seq))
        )
        predicted_seq_sample = predicted_seq.unsqueeze(dim=-2).repeat(1, 1, n_traj, 1)
        outputs = {
            'predicted_seq_sample': predicted_seq_sample,
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }
        return outputs, {'y': y}



    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        l, batch_size, _ = observations_seq.shape
        # external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]

        train_pred_len = int(len(external_input_seq) / 2)
        historical_input = external_input_seq[:train_pred_len]
        historical_ob = observations_seq[:train_pred_len]
        future_input_and_dt = external_input_seq[train_pred_len:]
        future_ob = observations_seq[train_pred_len:]

        outputs, memory_state = self.forward_posterior(historical_input, historical_ob, memory_state)
        posterior_x_seq = outputs['x_seq']

        # outputs, memory_state = self.forward_prediction(future_input, memory_state=memory_state, grad=True)

        # ts = dt2ts(dt[train_pred_len:])
        # ys = self.forward(ts, repeat_tail(future_input), repeat_tail(future_ob), memory_state['hn'], batch_size)  # repeat_tail：(241, 512, 1) -> (240, 512, 1)
        ys_outputs, y = self.forward_prediction(future_input_and_dt, memory_state=memory_state, grad=True)
        ys = ys_outputs['predicted_seq']
        # ys = ys[:-1]

        predicted_seq = torch.cat([posterior_x_seq, ys], dim=0)

        # MSE
        return {
            'loss': F.mse_loss(predicted_seq, observations_seq),  # TODO: observations_seq[1:]
            'kl_loss': 0,
            'likelihood_loss': 0
        }

    def decode_observation(self, outputs, mode='sample'):

        """

              Args:
                  outputs:
                  mode: dist or sample

              Returns:
                  model为sample时，从分布采样(len,batch_size,observation)
                  为dist时，直接返回分布对象torch.distributions.MultivariateNormal

              方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
              """
        if mode == 'dist':
            observations_normal_dist = MultivariateNormal(
                outputs['state_mu'], logsigma2cov(outputs['state_logsigma'])
            )
            return observations_normal_dist
        elif mode == 'sample':
            return outputs['state_mu']
