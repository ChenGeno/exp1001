#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import torch


class BaseNormalizer:
    def normalize(self, x, normal_type):
        raise NotImplementedError

    def inverse(self, x, normal_type):
        raise NotImplementedError

class Normalizer(object):
    def __init__(self, mean, std):
        if isinstance(mean, list) or isinstance(mean, np.ndarray):
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
        else:
            self.mean = mean
            self.std = std

    def __call__(self, x, inverse=False):
        if inverse:
            return self.inverse(x)
        else:
            return self.normalize(x)

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif isinstance(x, np.ndarray):
            return (x - self.mean.numpy()) / self.std.numpy()
        else:
            raise NotImplementedError

    def inverse(self, x):
        # return x * self.std + self.mean
        if isinstance(x, torch.Tensor):
            return x * self.std.to(x.device) + self.mean.to(x.device)
        elif isinstance(x, np.ndarray):
            return x * self.std.numpy() + self.mean.numpy()
        else:
            raise NotImplementedError


class EnvNormalizer(BaseNormalizer):
    def __init__(self, u_mean, u_std, y_mean, y_std, c_mean, c_std):
        self.normalize_u = Normalizer(u_mean, u_std)
        self.normalize_y = Normalizer(y_mean, y_std)
        self.normalize_c = Normalizer(c_mean, c_std)
        self.u_size, self.y_size, self.c_size = len(u_mean), len(y_mean), len(c_mean)
        # self.ob_size_split = np.cumsum([0, self.y_size, self.y_size, self.c_size, self.u_size])
        self.name2size = {'y': self.y_size, 'u': self.u_size, 'c': self.c_size}

    def __call__(self, x, type, inverse=False):
        if type == 'u':
            return self.normalize_u(x, inverse)
        elif type == 'y':
            return self.normalize_y(x, inverse)
        elif type == 'c':
            return self.normalize_c(x, inverse)
        elif type == 'observation':
            return self(x, 'yyuc', inverse)
        else:
            res = []
            cur = 0
            for c in type:
                res.append(self(x[..., cur:cur+self.name2size[c]], c, inverse))
                cur += self.name2size[c]
            if isinstance(x, torch.Tensor):
                return torch.cat(res, dim=-1)
            elif isinstance(x, np.ndarray):
                return np.concatenate(res, axis=-1)
            else:
                raise NotImplementedError

    def normalize(self, x, normal_type):
        return self(x, normal_type, inverse=False)

    def inverse(self, x, normal_type):
        return self(x, normal_type, inverse=True)


class NoNormalizer(BaseNormalizer):
    @staticmethod
    def normalize(x, *args, **kwargs):
        return x

    @staticmethod
    def inverse(x, *args, **kwargs):
        return x
