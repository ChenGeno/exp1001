#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
from multiprocessing import Process
import json
import torch
from Control_Exp1001.demo.thickener.hdp import HDP
from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.demo.thickener.adhdp_make import adhdp
import random
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
penalty_para = {
    #"weight_matrix": [0, 0.002],
    "weight_matrix": [0, 0.004],
    #"S": [0.00001, 0.00008],
    "S": [0.0001, 0.0008],
}
thickner_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    "noise_type": 2,
    #'time_length':40
}
mse_vi_pre=[]
mse_vi_sample_pre=[]
mse_vi=[]
mse_vi_sample=[]
def test_model_hidden():

    env = Thickener(noise_in=True)
    env.reset()
    loss_list = []
    hid_size_list = []
    for hidden_size in range(6,30,2):
        controller = HDP(
            replay_buffer = None,
            u_bounds=env.u_bounds,
            env=env,
            predict_training_rounds=10000,
            gamma=0.6,

            batch_size = 1,
            predict_batch_size=32,

            model_nn_error_limit = 0.00008,
            critic_nn_error_limit = 0.9,
            actor_nn_error_limit = 0.1,

            actor_nn_lr = 0.003,
            critic_nn_lr = 0.2,
            model_nn_lr = 0.01,

            indice_y = None,
            indice_y_star = None,
            indice_c=None,
            hidden_model = hidden_size,
            hidden_critic = 10,
            hidden_actor = 10,
            predict_epoch=40,
        )
        hid_size_list.append(hidden_size)
        controller.train_identification_model()
        loss = controller.cal_predict_mse(test_rounds=3000)
        loss_list.append(loss)

    plt.plot(hid_size_list, loss_list)

    plt.legend(['loss in test'])
    plt.show()


def run_hdp(rounds=1000,seed=random.randint(0,1000000),name='HDP', predict_round=800):


    hdp_para={
        'gamma':0.9
    }
    print('seed :',seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    from Control_Exp1001.demo.thickener.hdp_maker import new_hdp
    hdp = new_hdp(predict_round=predict_round, **hdp_para)
    penalty = Quadratic(**penalty_para)
    env_hdp = Thickener(
                    penalty_calculator=penalty,
                    random_seed=seed,
                    **thickner_para,
                    )


    res1 = OneRoundExp(controller=hdp, env=env_hdp,max_step=rounds, exp_name=name).run()


    return res1



    #controller.test_predict_model(test_rounds=100)

def run_adhdp(rounds=1000,seed=random.randint(0,1000000)):


    print('seed :',seed)
    random.seed(seed)
    np.random.seed(seed)
    penalty = Quadratic(**penalty_para)
    env_adhdp = Thickener(
            penalty_calculator=penalty,
            **thickner_para,
            random_seed=seed,
    )


    env_adhdp.reset()
    res1 = OneRoundExp(controller=adhdp, env=env_adhdp,max_step=rounds, exp_name='ADHDP').run()

    eval_res = OneRoundEvaluation(res_list=[res1])
    eval_res.plot_all()


def run_hdp_sample(rounds=1000,seed=random.randint(0,1000000)):


    print('seed :',seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print('hdp_sample')
    from Control_Exp1001.demo.thickener.hdp_sample_maker import hdp_sample
    penalty = Quadratic(**penalty_para)
    env_hdp = Thickener(
            penalty_calculator=penalty,
            random_seed=seed,
            **thickner_para,
    )

    res1 = OneRoundExp(controller=hdp_sample, env=env_hdp,max_step=rounds, exp_name='HDP_sample').run()

    return res1

    #controller.test_predict_model(test_rounds=100)


def run_vi(rounds=1000,seed=random.randint(0,1000000),name='VI',capacity=2,
           predict_round=3000,u_optim='adam'):

    print('seed :',seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    from Control_Exp1001.demo.thickener.vi_maker import new_vi
    vi = new_vi(capacity=capacity,predict_round=predict_round,u_optim=u_optim)
    penalty = Quadratic(**penalty_para)
    env_vi = Thickener(
                    penalty_calculator=penalty,
                    random_seed=seed,
                    **thickner_para,
                    )

    mse_vi_pre.append(vi.con_predict_mse)
    res1 = OneRoundExp(controller=vi, env=env_vi,max_step=rounds, exp_name=name).run()
    print(name,':',vi.u_iter_times*1.0/rounds)

    return res1

def run_vi_sample(rounds=1000,seed=random.randint(0,1000000),name='VI_sample',capacity=2,
                  predict_round=3000):

    print('seed :',seed)
    torch.manual_seed(seed)
    from Control_Exp1001.demo.thickener.vi_sample_maker import new_vi_sample
    vi_sample = new_vi_sample(capacity=capacity, predict_round=predict_round)
    penalty = Quadratic(**penalty_para)
    env_vi_sample = Thickener(
        penalty_calculator=penalty,
        random_seed=seed,
        **thickner_para,
    )
    res1 = OneRoundExp(controller=vi_sample, env=env_vi_sample,max_step=rounds, exp_name=name).run()

    mse_vi_sample_pre.append(vi_sample.con_predict_mse)
    return res1

def compare_hdp_hdpsample():
    print('hdp and sample')
    round = 1000
    rand_seed = np.random.randint(0,10000000)
    res_hdp = run_hdp(rounds=round,seed=rand_seed)
    res_hdp_sample = run_hdp_sample(rounds=round,seed=rand_seed)
    eval_res = OneRoundEvaluation(res_list=[res_hdp, res_hdp_sample])
    eval_res.plot_all()


def hdp_only():
    print('hdp only')
    round = 400
    rand_seed = np.random.randint(0,10000000)
    rand_seed = 7880643
    res_hdp = run_hdp(rounds=round,seed=rand_seed)
    eval_res = OneRoundEvaluation(res_list=[res_hdp])
    eval_res.plot_all()

def hdp_sample_only():
    print('hdp_sample only')
    round = 400
    rand_seed = np.random.randint(0,10000000)
    res_hdp = run_hdp(rounds=round,seed=rand_seed)
    eval_res = OneRoundEvaluation(res_list=[res_hdp])
    eval_res.plot_all()

def hdp_five_times():

    print('hdp 5')
    res_list = []
    for t in range(5):
        round = 400
        rand_seed = np.random.randint(0,10000000)
        res_list.append(run_hdp(rounds=round,seed=rand_seed, name='hdp_'+str(t+1)),)
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()

def vi_compare_hdp():
    round = 400
    predict_round=800
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    rand_seed = 7717763
    res_list.append(run_vi(rounds=round,seed=rand_seed, name='VI', predict_round=predict_round))
    res_list.append(run_hdp(rounds=round,seed=rand_seed, name='HDP', predict_round=predict_round))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()



def vi_test():
    exp_round = 3
    res_list = []
    for t in range(exp_round):
        round = 400
        rand_seed = np.random.randint(0,10000000)
        res_list.append(run_vi(rounds=round,seed=rand_seed, name='VI_'+str(t+1)),)
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()

def vi_compare_sample():
    exp_round = 1
    res_list = []
    predict_round=800
    rand_seed = np.random.randint(0,10000000)


    for t in range(exp_round):
        round = 400
        res_list.append(run_vi(rounds=round,seed=rand_seed, name='VI', predict_round=predict_round),)

    for t in range(exp_round):
        round = 400
        res_list.append(run_vi_sample(rounds=round,seed=rand_seed, name='VI_sample', predict_round=predict_round),)



    eval_res = OneRoundEvaluation(res_list=res_list)
    mse_dict = eval_res.plot_all()

    mse_vi.append(mse_dict['VI'])
    mse_vi_sample.append(mse_dict['VI_sample'])


def vi_diff_capacity(capacity_list=None):

    if capacity_list is None:
        capacity = range(1,12,3)

    predict_round=800
    res_list = []

    rand_seed = np.random.randint(0,10000000)

    rand_seed = 7717763
    for capacity in capacity_list:
        round = 400
        res_list.append(run_vi(rounds=round,seed=rand_seed,capacity=capacity,
                               name='Replay: '+str(capacity),predict_round=predict_round))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()


def vi_compard_simple_multi_times():
    for i in range(3):
        vi_compare_sample()
        #vi_compare_sample()
        os.remove('training_data_800.json')

    print(mse_vi_sample)
    print(mse_vi)
    print(mse_vi_sample_pre)
    print(mse_vi_pre)
    fig1 = plt.figure()
    colors = ['b', 'g', 'r', 'orange']
    label = ['VI_sample control', 'VI control','VI_sample predict','VI predict control']
    for id, array in enumerate([mse_vi_sample, mse_vi, mse_vi_sample_pre, mse_vi_pre]):
        array = np.array(array)
        std = np.std(array)
        mean = np.mean(array)
        array=(array-mean)/std
        plt.scatter(np.arange(0,10,10), array, c=colors[id], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)
    ax = fig1.gca()

    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.xlabel('Time')
    plt.ylabel('Price')
    # added this to get the legend to work
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels=label, loc='upper right')
    plt.title('Relations between MSE in control and MSE in forecast')
    plt.show()

    from scipy.stats import pearsonr
    print('pearson:', pearsonr(mse_vi_pre+mse_vi_sample_pre, mse_vi+mse_vi_sample))


def vi_optim_compare():
    exp_round = 1
    res_list = []
    predict_round=800
    rand_seed = np.random.randint(0,10000000)

    opt_list = [ 'adagrad','RMSprop', 'adam', 'sgd']
    opt_list = [ 'adam', 'sgd']
    for opt_name in opt_list:
        round = 400
        res_list.append(run_vi(rounds=round,seed=rand_seed, name='VI_'+opt_name,
                                      predict_round=predict_round, u_optim=opt_name),)

    eval_res = OneRoundEvaluation(res_list=res_list)
    mse_dict = eval_res.plot_all()


if __name__ == '__main__':

    # HDP算法单独运行
    # run_hdp()
    # HDP算法仅测试预测网络性能
    # test_model_hidden()
    # ADHDP算法单独运行
    # run_adhdp()
    # run_hdp_sample()
    # compare_hdp_hdpsample()
    # vi_diff_capacity()
    # for i in range(1):
    #     vi_compare_hdp()
    #     #vi_compare_sample()
    #     os.remove('training_data_800.json')

    # vi_compare_hdp()
    # vi_optim_compare()
    for i in range(1):
        vi_diff_capacity(capacity_list=[1,2,5])
    # vi_compare_hdp()



