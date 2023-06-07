#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from .common import (PreProcess, MLP, logsigma2cov,
                     DiagMultivariateNormal as MultivariateNormal)
from .func import zeros_like_with_shape
from . import BaseModel


class NN(nn.Module, BaseModel):

    def __init__(self, input_size, observations_size, h_size=16, num_layers=1):

        super(NN, self).__init__()

        self.h_size = h_size
        self.input_size = input_size
        self.observations_size = observations_size

        self.process_u = PreProcess(input_size, h_size)
        self.process_x = PreProcess(observations_size, h_size)
        self.predictor = MLP(2*h_size, 3*h_size, observations_size, num_layers)

    def _forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        outputs = {
            'state_mu': observations_seq,
            'state_logsigma': -torch.ones_like(observations_seq) * float('inf'),
        }
        pred_next_ob = self.predictor(
            torch.cat([self.process_u(external_input_seq[-1]), self.process_x(observations_seq[-1])], dim=-1)
        )

        return outputs, {'yn': pred_next_ob}

    def _forward_prediction(self, external_input_seq, n_traj=1, memory_state=None):

        l, batch_size, _ = external_input_seq.size()

        external_input_seq_embed = self.process_u(external_input_seq)

        yn = zeros_like_with_shape(external_input_seq, (batch_size, self.observations_size)
                                   ) if memory_state is None else memory_state['yn']

        predicted_seq = []
        for t in range(l):
            predicted_seq.append(yn)
            yn = self.predictor(torch.cat([external_input_seq_embed[t], self.process_x(yn)], dim=-1))
        predicted_seq = torch.stack(predicted_seq, dim=0)
        predicted_dist = MultivariateNormal(
            predicted_seq, torch.diag_embed(torch.zeros_like(predicted_seq))
        )

        predicted_seq_sample = predicted_seq.unsqueeze(dim=-2).repeat(1, 1, n_traj, 1)
        outputs = {
            'predicted_seq_sample': predicted_seq_sample,
            'predicted_dist': predicted_dist,
            'predicted_seq': predicted_seq
        }
        return outputs, {'yn': yn}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """
        Teacher forcing模式训练
        :param external_input_seq:
        :param observations_seq:
        :param memory_state:
        :return:
        """

        pred_ob = self.predictor(
            torch.cat([self.process_u(external_input_seq[:-1]), self.process_x(observations_seq[:-1])], dim=-1)
        )
        # outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)
        # predicted_seq = outputs['x_seq']
        return {
            'loss': F.mse_loss(pred_ob, observations_seq[1:]),
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

