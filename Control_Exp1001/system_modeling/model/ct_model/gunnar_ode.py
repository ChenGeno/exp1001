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
                 dt=1e-2, rtol=1e-3, atol=1e-3, method='euler', adaptive=False):
        super(GunnarODE, self).__init__()

        u_size -= 1 # The last dimension of input variable represents dt
        self.y_size, self.u_size, self.h_size, self.inter = y_size, u_size-1, h_size, inter
        self.dt, self.rtol, self.atol, self.method, self.adaptive = dt, rtol, atol, method, adaptive

        self.process_u = PreProcess(u_size, h_size)
        # self.process_x = PreProcess(y_size, h_size)

        # TODO：batchsize
        self.cde_func = NCDEFunc(y_size, u_size, batch_size=512)

    def forward(self, ts, us, ys, batch_size):

        # 预处理，维度适应, 从(l, batch, v) -> (batch, l, v)
        ts = ts.permute(1, 0, 2)
        us = us.permute(1, 0, 2)
        ys = ys.permute(1, 0, 2)

        # TODO: 定义X
        x = torch.cat([ts, us], dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)

        # TODO: 定义y0（z0）
        z0 = ys[:, 0, :]

        ys = torchcde.cdeint(X=X,
                             func=self.cde_func,
                             z0=z0,
                             t=X.interval)

        return ys


    def _forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """
        训练时：估计隐变量后验分布，并采样，用于后续计算模型loss
        测试时: 为后续执行forward_prediction计算memory_state(h, rnn_hidden)
        Args:
            external_input_seq: 系统输入序列(进出料浓度、流量) (len, batch_size, input_size)
            observations_seq: 观测序列(泥层压强) (len, batch_size, observations_size)
            memory_state: 模型支持长序列预测，之前forward过程产生的记忆信息压缩在memory_state中

        Returns:

        """

        # external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        # l, batch_size, _ = external_input_seq.size()
        y0 = memory_state['y'] if memory_state is not None else None
        # ts = dt2ts(dt)

        outputs = {
            'state_mu': observations_seq,
            'state_logsigma': -torch.ones_like(observations_seq) * float('inf'),
        }
        # todo
        pred_next_ob = 0
        return outputs, {'yn': pred_next_ob}

    def _forward_prediction(self, external_input_seq, n_traj=16, memory_state=None):
        l, batch_size, _ = external_input_seq.size()
        external_input_seq = external_input_seq.repeat(1, n_traj, 1)
        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        ts = dt2ts(dt)
        device = external_input_seq.device
        y0 = memory_state['y'] if memory_state is not None else None
        y0 = y0.repeat(n_traj, 1)
        ys = self.sample_p(ts, repeat_tail(external_input_seq), batch_size=batch_size, y0=y0, eps=0, bm=None)
        ys, y = ys[:-1], ys[-1, :batch_size]
        mean, logsigma = self.decoder(ys)

        predicted_seq_sample = normal_differential_sample(
            MultivariateNormal(mean, logsigma2cov(logsigma))
        )
        predicted_seq_sample = rearrange(predicted_seq_sample, 'l (n b) d -> l n b d', n=n_traj, b=batch_size).permute(
            0, 2, 1, 3).contiguous()
        predicted_seq = torch.mean(predicted_seq_sample, dim=2)
        predicted_dist = MultivariateNormal(
            predicted_seq, torch.diag_embed(predicted_seq_sample.var(dim=2))
            # 此处如何生成分布(如何提取均值和方差)
        )
        outputs = {
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq,
            'predicted_seq_sample': predicted_seq_sample
        }
        return outputs, {'y': y}


    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        l, batch_size, _ = observations_seq.shape
        external_input_seq, dt = external_input_seq[..., :-1], external_input_seq[..., -1:]
        ts = dt2ts(dt)
        ys = self.forward(ts, repeat_tail(external_input_seq), repeat_tail(observations_seq), batch_size)  # repeat_tail：(241, 512, 1) -> (240, 512, 1)
        ys = ys[:-1]

        # MSE
        return {
            'loss': F.mse_loss(ys, observations_seq[1:]),
            'kl_loss': 0,
            'likelihood_loss': 0
        }

    def decode_observation(self, outputs, mode='sample'):

        """
        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean, logsigma = self.decoder(outputs['sampled_state'])
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'mean':
            return mean
        elif mode == 'sample':
            return observations_normal_dist.sample()
