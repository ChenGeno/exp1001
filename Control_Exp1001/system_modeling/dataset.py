#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from .common import onceexp, detect_download, obs2data_paths
import hydra
import pandas as pd


class WesternDataset(Dataset):
    def __init__(self, df_list, length=1000, step=5, dilation=2):
        """

        Args:
            df_list:
            length:
            step: 数据segment切割窗口的移动步长
            dilation: 浓密机数据采样频率(1 min)过高，dilation表示数据稀释间距
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        # 每个column对应的数据含义 ['c_in','c_out', 'v_out', 'v_in', 'pressure']
        self.used_columns = ['4', '11', '14', '16', '17']
        self.length = length
        self.dilation = dilation

        for df in df_list:
            # 一个文件是一个df，所有文件构成df_list
            df_split_all = df_split_all + self.split_df(df[self.used_columns])
        for i, df in enumerate(df_split_all):
            for j in range(0, df.shape[0] - length * dilation + 1, step):
                # 建立滑动窗口index集合
                begin_pos_pair.append((i, j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all
        self.df_split_all = self.normalize(self.df_split_all)

    def normalize(self, df_all_list):
        df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
        mean = df_all.mean()
        std = df_all.std()
        return [(df - mean) / std for df in df_all_list]

    def split_df(self, df):
        """
        将存在空值的位置split开
        Args:
            df:
        Returns: list -> [df1,df2,...]
        """
        df_list = []
        split_indexes = list(
            df[df.isnull().T.any()].index
        )
        split_indexes = [-1] + split_indexes + [df.shape[0]]
        for i in range(len(split_indexes) - 1):
            if split_indexes[i + 1] - split_indexes[i] - 1 < self.length:
                continue

            new_df = df.iloc[split_indexes[i] + 1:split_indexes[i + 1]]
            assert new_df.isnull().sum().sum() == 0
            df_list.append(new_df)
        return df_list

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        data_array = np.array(self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation], dtype=np.float32)
        data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        c_in, c_out, v_out, v_in, pressure = [np.squeeze(x, axis=1) for x in np.hsplit(data_array, 5)]

        v_in = v_in * 0.05
        v_out = v_out * 0.05

        external_input = np.stack(
            [
                c_in * c_in * c_in * v_in - c_out * c_out * c_out * v_out,
                c_in * c_in * v_in - c_out * c_out * v_out,
                c_in * v_in - c_out * v_out,
                v_in - v_out,
                v_in,
                v_out,
                c_in,
                c_out
            ],
            axis=1)
        observation = pressure
        return external_input, np.expand_dims(observation, axis=1)


class WesternDataset_1_4(WesternDataset):
    """
    1进4出的，为了跑实验写的
    """
    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        data_array = np.array(self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation], dtype=np.float32)
        data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        c_in, c_out, v_out, v_in, pressure = [np.squeeze(x, axis=1) for x in np.hsplit(data_array, 5)]

        external_input = v_out

        observation = np.stack(
            [
                c_out,
                c_in,
                v_in,
                pressure
            ],
            axis=1)

        return np.expand_dims(external_input, axis=1), observation


class CstrDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df[['1', '2']], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return np.expand_dims(data_in, axis=1), data_out


class NLDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['input','output']
        self.df = df
        self.used_columns = ['u', 'y']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df['u'], dtype=np.float32)
        data_out = np.array(data_df['y'], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class IBDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []
        self.mean = 0.0
        self.std = 0.0
        # 每个column对应的数据含义 ['delta_v', 'delta_g', 'delta_h','f','c', 'reward']
        self.df = df
        self.used_columns = ['delta_v', 'delta_g', 'delta_h', 'v', 'g', 'h', 'f', 'c', 'reward']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        self.mean = df.mean()
        self.std = df.std()

        return (df - self.mean) / self.std

    def normalize_record(self):
        mean = self.mean
        std = self.std
        return [mean, std]

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df[['delta_v', 'delta_g', 'delta_h']], dtype=np.float32)
        data_out = np.array(data_df[['v', 'g', 'h', 'f', 'c', 'reward']], dtype=np.float32)
        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class WindingDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2', '3', '4', '5', '6']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df[['0', '1', '2', '3', '4']], dtype=np.float32)
        data_out = np.array(data_df[['5', '6']], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class WesternConcentrationDataset(Dataset):
    def __init__(self, df_list, length=1000, step=5, dilation=2):
        """

        Args:
            df_list:
            length:
            step: 数据segment切割窗口的移动步长
            dilation: 浓密机数据采样频率(1 min)过高，dilation表示数据稀释间距
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        # 每个column对应的数据含义 ['c_in','c_out', 'v_out', 'v_in', 'pressure']
        self.used_columns = ['4', '5', '7', '11', '14', '16', '17']
        self.length = length
        self.dilation = dilation

        for df in df_list:
            df_split_all = df_split_all + self.split_df(df[self.used_columns])
        for i, df in enumerate(df_split_all):
            for j in range(0, df.shape[0] - length * dilation + 1, step):
                begin_pos_pair.append((i, j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all
        self.df_split_all = self.normalize(self.df_split_all)

    def normalize(self, df_all_list):
        df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
        mean = df_all.mean()
        std = df_all.std()
        return [(df - mean) / std for df in df_all_list]

    def split_df(self, df):
        """
        将存在空值的位置split开
        Args:
            df:
        Returns: list -> [df1,df2,...]
        """
        df_list = []
        split_indexes = list(
            df[df.isnull().T.any()].index
        )
        split_indexes = [-1] + split_indexes + [df.shape[0]]
        for i in range(len(split_indexes) - 1):
            if split_indexes[i + 1] - split_indexes[i] - 1 < self.length:
                continue

            new_df = df.iloc[split_indexes[i] + 1:split_indexes[i + 1]]
            assert new_df.isnull().sum().sum() == 0
            df_list.append(new_df)
        return df_list

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        # data_array = np.array(self.df_split_all[df_index].iloc[pos:pos+self.length*self.dilation], dtype=np.float32)
        # data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]

        # data_in = np.array(data_array[['4','5','7','14','16']], dtype=np.float32)
        # data_out = np.array(data_array[['11', '17']], dtype=np.float32)

        data_df = self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation]

        def choose_and_dilation(df, length, dilation, indices):
            return np.array(
                df[indices], dtype=np.float32
            )[np.arange(length) * dilation]

        data_in = choose_and_dilation(data_df, self.length, self.dilation, ['4', '5', '7', '14', '16'])
        data_out = choose_and_dilation(data_df, self.length, self.dilation, ['11', '17'])

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class SoutheastThickener(Dataset):
    def __init__(self, data, length=90, step=5, dilation=1, dataset_type=None, ratio=None, io=None, seed=0,
                 smooth_alpha=0.3):
        """

        Args:
            data: data array
            length: history + predicted
            step:  size of moving step
            dilation:
            dataset_type:  train, val, test
            ratio:  default: 0.6, 0.2, 0.2
            io: default 4-1
            seed: default 0
        """

        if not isinstance(seed, int):
            seed = 0

        if dataset_type is None:
            dataset_type = 'train'

        if ratio is None:
            ratio = [0.6, 0.2, 0.2]

        if io is None:
            io = '4-1'

        if io == '4-1':
            # 进料浓度、出料浓度、进料流量、出料流量 -> 泥层压力
            self.io = [[0, 1, 2, 3], [4]]
        elif io == '3-2':
            # 进料浓度、进料流量、出料流量 -> 出料浓度 、泥层压力
            self.io = [[0, 2, 3], [1, 4]]
        else:
            raise NotImplementedError()

        # region old version
        # data = np.array(data, dtype=np.float32)
        # endregion

        # region new version
        def iter(data):
            for k, v in data.item().items():
                for i in range(0, v.shape[0] - length * dilation + 1, step):
                    yield v[i:i+length]
        data = np.stack([x for x in iter(data)], axis=0)
        data = np.array(data, dtype=np.float32)
        # endregion

        self.smooth_alpha = smooth_alpha

        for _ in self.io[1]:
            # data.shape (N, 90, 5)
            data[:, :, int(_)] = onceexp(data[:, :, int(_)].transpose(), self.smooth_alpha).transpose()

        data, self.mean, self.std = self.normalize(data)

        data = data[::step]
        L = data.shape[0]

        train_size, val_size = int(L*ratio[0]), int(L*ratio[1])
        test_size = L - train_size - val_size

        d1, d2, d3 = torch.utils.data.random_split(data, (train_size, val_size, test_size),
                                                   generator=torch.Generator().manual_seed(seed))
        if dataset_type == 'train':
            self.reserved_dataset = d1
        elif dataset_type == 'val':
            self.reserved_dataset = d2
        elif dataset_type == 'test':
            self.reserved_dataset = d3
        else:
            raise AttributeError()

        self.dilation = dilation
        self.step = step

    def normalize(self, data):
        mean = np.mean(data, axis=(0, 1))
        std = np.std(data, axis=(0, 1))
        return (data - mean) / std, mean, std

    def __len__(self):
        return len(self.reserved_dataset)

    def __getitem__(self, item):
        data_tuple = self.reserved_dataset.__getitem__(item)
        # data_tuple = self.reserved_data[item * self.step]
        data_in, data_out = [data_tuple[:, self.io[_]] for _ in range(2)]

        return data_in, data_out





# PR-SSM Dataset: actuator, ballbeam, drive, dryer, gas_furnace
class ActuatorDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['u', 'p']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['u'], dtype=np.float32)
        data_out = np.array(data_df['p'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class BallbeamDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df['1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class DriveDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['u1', 'z1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['u1'], dtype=np.float32)
        data_out = np.array(data_df['z1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class DryerDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df['1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class GasFurnaceDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df['1'], dtype=np.float32)

        return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)


class SarcosArmDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['0', '1', '2', '3', '4', '5', '6',
                             '21', '22', '23', '24', '25', '26', '27']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df[['21', '22', '23', '24', '25', '26', '27']], dtype=np.float32)
        data_out = np.array(data_df[['0', '1', '2', '3', '4', '5', '6']], dtype=np.float32)

        return data_in, data_out

class Thickener_Simulation(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['y_h', 'y_c', 'u_fu', 'u_ff', 'c_fi', 'c_ci']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df[['u_fu', 'u_ff']], dtype=np.float32)
        data_out = np.array(data_df[['y_h', 'y_c', 'c_fi', 'c_ci']], dtype=np.float32)

        return data_in, data_out

class Thickener_Rake(Dataset):
    def __init__(self, df, length=1000, step=5):
        begin_pos = []

        # 每个column对应的数据含义 ['in','out']
        self.df = df
        self.used_columns = ['feed_c', 'out_c', 'feed_f', 'out_f', 'pressure', 'rake_1', 'rake_2', 'rake_3', 'rake_4']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        data_in = np.array(data_df[['feed_c', 'out_c', 'feed_f', 'out_f', 'pressure']], dtype=np.float32)
        data_out = np.array(data_df[['rake_1', 'rake_2', 'rake_3', 'rake_4']], dtype=np.float32)

        return data_in, data_out

# class Creep_Curve(Dataset):
#     """
#     高温蠕变曲线
#     """
#     def __init__(self, df_list, length):
#         # 每个column对应的数据含义
#         self.used_columns = ['t', 'creepage', 'beta']
#         self.length = length
#         # self.dilation = dilation
#
#     def normalize(self, df_all_list):
#         df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
#         mean = df_all.mean()
#         std = df_all.std()
#         return [(df - mean) / std for df in df_all_list]
#
#     def __len__(self):
#         return len(self.begin_pos_pair)
#
#     def __getitem__(self, item):
#         return 0


def prepare_training_dataset(args, logging):
    access_key = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'data', 'AccessKey.csv'))
    if args.dataset.type.startswith('west'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data', 'west', 'data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/west')
        if not os.path.exists(base):
            os.mkdir(base)
        # 检测数据集路径，如果本地没有数据自动下载
        data_paths = obs2data_paths(objects,
                                     base,
                                     'http://oss-cn-beijing.aliyuncs.com',
                                     'west-part-pressure',
                                     access_key['AccessKey ID'][0],
                                     access_key['AccessKey Secret'][0]
                                     )
        data_csvs = [pd.read_csv(path) for path in data_paths]
        dataset_split = [0.6, 0.2, 0.2]
        # 训练测试集的比例划分
        train_size, val_size, test_size = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        if args.dataset.type.endswith('1_4'):
            train_dataset = WesternDataset_1_4(data_csvs[:train_size],
                                               args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                               dilation=args.dataset.dilation)
            val_dataset = WesternDataset_1_4(data_csvs[train_size:train_size + val_size],
                                             args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                             dilation=args.dataset.dilation)
        else:
            train_dataset = WesternDataset(data_csvs[:train_size],
                                           args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                           dilation=args.dataset.dilation)
            val_dataset = WesternDataset(data_csvs[train_size:train_size + val_size],
                                         args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window,
                                         dilation=args.dataset.dilation)
    elif args.dataset.type == 'west_con':
        data_dir = os.path.join(hydra.utils.get_original_cwd(), 'data/west_con')
        data_csvs = [pd.read_csv(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
        dataset_split = [0.6, 0.2, 0.2]
        train_size, val_size, test_size = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        train_dataset = WesternConcentrationDataset(data_csvs[:train_size],
                                                    args.dataset.history_length + args.dataset.forward_length,
                                                    step=args.dataset.dataset_window,
                                                    dilation=args.dataset.dilation)
        val_dataset = WesternConcentrationDataset(data_csvs[train_size:train_size + val_size],
                                                  args.dataset.history_length + args.dataset.forward_length,
                                                  step=args.dataset.dataset_window,
                                                  dilation=args.dataset.dilation)
    elif args.dataset.type.startswith('cstr'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/cstr/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/cstr')
        if not os.path.exists(base):
            os.mkdir(base)
        _ = obs2data_paths(objects, base)
        train_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('actuator'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/actuator/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/actuator')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = ActuatorDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = ActuatorDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('ballbeam'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/ballbeam/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/ballbeam')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = BallbeamDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = BallbeamDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('drive'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/drive/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/drive')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = DriveDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = DriveDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('dryer'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/dryer/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/dryer')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = DryerDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = DryerDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('gas_furnace'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/gas_furnace/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/gas_furnace')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = GasFurnaceDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = GasFurnaceDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('sarcos'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/sarcos/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/sarcos')
        if not os.path.exists(base):
            os.mkdir(base)

        train_dataset = SarcosArmDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = SarcosArmDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('nl'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/nl/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/nl')
        if not os.path.exists(base):
            os.mkdir(base)
        train_dataset = NLDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = NLDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('ib'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/ib/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/ib')
        if not os.path.exists(base):
            os.mkdir(base)
        train_dataset = IBDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = IBDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        scale = train_dataset.normalize_record()
    elif args.dataset.type.startswith('thickener_sim'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/thickener_sim/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/thickener_sim')
        if not os.path.exists(base):
            os.mkdir(base)
        train_dataset = Thickener_Simulation(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = Thickener_Simulation(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('thickener_rake'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/thickener_rake/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/thickener_rake')
        if not os.path.exists(base):
            os.mkdir(base)
        train_dataset = Thickener_Rake(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = Thickener_Rake(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('winding'):
        objects = pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), 'data/winding/data_url.csv')
        )
        base = os.path.join(hydra.utils.get_original_cwd(), 'data/winding')
        if not os.path.exists(base):
            os.mkdir(base)
        _ = detect_download(objects,
                            base,
                            'http://oss-cn-beijing.aliyuncs.com',
                            'io-system-data',
                            access_key['AccessKey ID'][0],
                            access_key['AccessKey Secret'][0]
                            )
        train_dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('southeast'):

        data = np.load(os.path.join(hydra.utils.get_original_cwd(), args.dataset.data_path), allow_pickle=True)
        train_dataset = SoutheastThickener(data,
                                           length=args.dataset.history_length + args.dataset.forward_length,
                                           step=args.dataset.dataset_window,
                                           dataset_type='train', io=args.dataset.io,
                                           smooth_alpha=args.dataset.smooth_alpha
                                           )

        val_dataset = SoutheastThickener(data,
                                         length=args.dataset.history_length + args.dataset.forward_length,
                                         step=args.dataset.dataset_window,
                                         dataset_type='val', io=args.dataset.io, smooth_alpha=args.dataset.smooth_alpha
                                         )

    return train_dataset, val_dataset
    # # 构建dataloader

    # collate_fn = None if not args.ct_time else CTSample(args.sp, args.base_tp, evenly=args.sp_even).batch_collate_fn
    # train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size,
    #                           shuffle=True, num_workers=args.train.num_workers, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=args.train.batch_size, shuffle=False,
    #                         num_workers=args.train.num_workers, collate_fn=collate_fn)

    # return train_loader, val_loader
