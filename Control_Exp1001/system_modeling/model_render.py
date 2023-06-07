#!/usr/bin/python
# -*- coding:utf8 -*-
import os.path
import sys
# append the dir of current file to sys.path
import shlex, hydra, sys
import torch
from torch import nn
from .train import train
from .model.generate_model import generate_model


def unsqueeze(func):
    def warp(*args, **kwargs):
        new_args = list(args)
        for i in range(len(new_args)):
            if isinstance(new_args[i], torch.Tensor) and len(new_args[i].size()) != 3:
                while len(new_args[i].size()) != 3:
                    new_args[i] = new_args[i].unsqueeze(dim=0)
        return func(*new_args, **kwargs)
    return warp


class SystemModel(nn.Module):
    """
    创建SystemModel类的目的是构建system modeling的访问接口：

    背景：
    system modeling 中假设状态的转移过程为 st, at --> st+1，而非 st-1, at --> st ，明确这一点对于理解为什么要开发本部分代码非常重要 !!!
    即system modeling中的posterior和prediction含义如下：
    posterior:
        Input:
            observations_seq:   s1, s2, s3
            external_input_seq: a1, a2, a3
            memory_state:       m1
        Output:
            outputs:            t=1到t=3的重构信息
            memory_state:                   m4: m4代表 t=4 位置的memory state


    prediction:
            external_input_seq:             a4, a5, a6
            memory_state:                   m4
        Output:
            outputs:                        s4, s5, s6      : 均为预测值/预测分布
            memory_state:                               m7  : m5代表 t=5 位置的memory state

   system modeling 中的推理/预测形式会存在两个问题，导致应用于Model-based RL时使用起来非常不方便：
   1. 对于当前时刻t，已知状态st, 但是at还是未知的(可能需要MPC算法预测不同的at产生的效益值，然后选择最优的at)，
   此时无法用posterior将st的信息加入到memory state中，因为posterior的输入中必须包含at
   2. 对于一个基本的mdp单步预测(假设t=3) s3, a3 --> s4 ，想要得到s4只能通过prediction(a4,m4)实现，但是a4是未知的
   （a4是多少对s4不产生影响）

    为了解决这两个问题，我们需要构造本SystemModel以对system modeling的推理/预测过程进行一定的修改：

    原始的函数(省去forward)   ->   本类中的函数
    posterior              ->   encode
    prediction             ->   predict


    ------------------------------------------------------------------
    encoder函数： 对于第1个问题，SystemModel中引入last_observation以存储孤立的observation。类中的encode方法支持
    observations_seq的长度比external_input_seq的长度大1，即支持对s1:sn+1, a1:an的编码。last_observation只是暂存最新的
    观测，用来跟预测过程predict进行配合，不会影响 self.memory_state，

    ------------------------------------------------------------------
    predict函数: 对于第2个问题，首先断言调用predict预测时last_observation是非空的，以s3, a3, a4 --> s4, s5为例，将预测过程分成步骤:

        前提假设：
            1. last_observation 为 s3 ,
            2. memory_state = m2 (即最后一次调用encode执行的是posterior(s2, a2))

        预测过程 s3, a3, a4 --> s4, s5 拆解为如下步骤
        1. tmp_state = posterior(s3, a3, memory_state)  # tmp_state = m3
        2. 构造虚拟 a5 = 0 # 因为prediction函数要求给定t=5时刻的a5，才会返回对应s5的预测值
        3. 执行预测[s4, s5], m6 = prediction([a4, a5], tmp_state) # 此处m6是没意义的，因为a5是认为构造的
        4. 如果希望用predict函数更新memoray state:
            new_memory_state = prediction([a4], tmp_state) # new_memory_state = m4
    --------------------------------------------------


    """
    def __init__(self, input_dim, output_dim, override_str=None, logging=None):
        super(SystemModel, self).__init__()
        if override_str is None:
            override_str = 'model=rssm'
        self.args = get_system_modeling_config(input_dim, output_dim, override_str)
        self.logging = print if logging is None else logging
        self.model = generate_model(self.args)
        self.memory_state = None
        self.last_observation = None

    def start_train(self, train_dataset, val_dataset=None, ckpt_path=None):

        from Control_Exp1001.utils import ChangePath
        val_dataset = train_dataset if val_dataset is None else val_dataset

        with ChangePath(ckpt_path):
            train(self.args, self.model, train_dataset, val_dataset, self.logging)

    @unsqueeze
    def encode(self, observations_seq, external_input_seq=None):
        len_obs, batch_size, _ = observations_seq.size()
        if external_input_seq is None:
            if len_obs == 1:
                self.last_observation = observations_seq[-1]
                return
            else:
                raise ValueError('external_input_seq is None, but length of observations_seq is not 1')
        else:
            if len_obs == external_input_seq.size(0) + 1:
                last_observation = observations_seq[-1]
                observations_seq = observations_seq[:-1]
            elif len_obs == external_input_seq.size(0):
                last_observation = None
            else:
                raise ValueError('Length of observations_seq must be equal to length of external_input_seq '
                                 'or length of external_input_seq + 1')

            _, self.memory_state = self.model.forward_posterior(external_input_seq, observations_seq, self.memory_state)
            self.last_observation = last_observation

    @unsqueeze
    def predict(self, external_input_seq, n_traj=1, update_memory_state=False, pred_type='sample', grad=True):
        """
        给定
        :param external_input_seq: L, bs, dim_out
        :param n_traj:
        :param update_memory_state:
        :param pred_type:
            sample: (L, bs, n_traj, dim_out)
            dist: A instance of MultivariateNormal. The shape of mean is  (L, bs, dim_out)
            one: Equivalent to sample with n_traj=1 and squeeze the dimension of n_traj
            all: A dict, the output of self.model.forward_prediction
        :return:
        """
        assert external_input_seq.size(0) > 0
        assert self.last_observation is not None

        outputs, tmp_memory_state = self.model.forward_posterior(
            external_input_seq[:1], self.last_observation.unsqueeze(dim=0), self.memory_state
        )
        external_input_seq_fake = torch.cat(
            [external_input_seq[1:], torch.zeros_like(external_input_seq[-1:])], dim=0
        )
        results_dict, _ = self.model.forward_prediction(
            external_input_seq_fake, n_traj, tmp_memory_state, grad=grad
        )
        if update_memory_state:
            _, self.memory_state = self.model.forward_prediction(
                external_input_seq[1:], n_traj, tmp_memory_state, grad=grad
            )

        if pred_type == 'sample':
            return results_dict['predicted_seq_sample']
        elif pred_type == 'dist':
            return results_dict['predicted_dist']
        elif pred_type == 'one':
            return results_dict['predicted_seq']
        else:
            return results_dict

    @unsqueeze
    def single_step_predict(self, x, u, n_traj=1, pred_type='sample', grad=True):
        """
        适用于低维空间的 st, at --> st+1
        :param x:
        :param u:
        :param return_memory_state:
        :param n_traj:
        :param pred_type:
        :return:
        """
        self.encode(x)
        return self.predict(u, n_traj=n_traj, update_memory_state=False, pred_type=pred_type, grad=grad)[0]

    def reset(self):
        self.memory_state = None

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def get_system_modeling_config(input_dim, output_dim, override_str=""):
    override_args = get_default_modeling_config_with_override_str(override_str)
    override_args.dataset.input_size = input_dim
    override_args.dataset.observation_size = output_dim
    override_args.dataset.type = 'Unknown'
    return override_args


def get_default_modeling_config_with_override_str(override_str):

    """
    将override_str作为命令行参数，从system modeling的config目录中加载配置项
    :param arg_str:
    :return:
    """

    construct_argv = ['main.py'] + shlex.split(override_str)
    sys.argv = construct_argv
    config = []

    @hydra.main(config_path="config", config_name="config.yaml")
    def generate_config(cfg):
        # config.append(cfg)
        config.append(cfg)
    generate_config()
    return config[0]


