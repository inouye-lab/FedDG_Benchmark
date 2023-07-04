from math import floor
from collections import OrderedDict
from numbers import Number
from typing import Callable, Optional
from torch.utils.data import DataLoader
from typing import Sequence

import logging
import operator
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import Subset
logger = logging.getLogger(__name__)



#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model
    

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def sign(self):
        return ParamDict({k: torch.sign(v) for k, v in self.items()})
    
    def ge(self, number):
        return ParamDict({k: torch.ge(v, number) for k, v in self.items()})
    
    def le(self, number):
        return ParamDict({k: torch.le(v, number) for k, v in self.items()})
    
    def gt(self, number):
        return ParamDict({k: torch.gt(v, number) for k, v in self.items()})
    
    def lt(self, number):
        return ParamDict({k: torch.lt(v, number) for k, v in self.items()})
    
    def abs(self):
         return ParamDict({k: torch.abs(v) for k, v in self.items()})

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

    def to(self, device):
        return ParamDict({k: v.to(device) for k, v in self.items()})

# class TransformSubset(Subset):
#     def __init__(self, dataset, indices, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
#         self.transform = transform
#         self.target_transform = target_transform
#         super().__init__(dataset, indices)

#     def __getitem__(self, idx):
#         if isinstance(idx, list):
#             sample, target = self.dataset[[self.indices[i] for i in idx]]
#         sample, target = self.dataset[self.indices[idx]]
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, target


# class DomainDataloader:
#     def __init__(self, datasets, batch_size, n_domains_per_batch, shuffle=True):
#         self.batch_size = batch_size
#         self.n_domains_per_batch = n_domains_per_batch
#         dataloaders = []
#         for _, dataset in datasets.items():
#             dataloaders.append(DataLoader(dataset, batch_size=batch_size//n_domains_per_batch, shuffle=shuffle))

#         self.dataloaders = dataloaders

#     def __iter__(self):
#         self.n_per_domain= np.array([len(loader) for loader in self.dataloaders])
#         return self

#     def __next__(self):
#         # print(self.n_per_domain)
#         # print(sum(self.n_per_domain))
#         nonzero_indices = np.nonzero(self.n_per_domain)[0]
#         # print(nonzero_indices)
#         # print("indices")
#         if len(nonzero_indices) <= 0:
#             # print("reset!")
#             self.n_per_domain = np.array([len(loader) for loader in self.dataloaders])
#             for dataloader in self.dataloaders:
#                 dataloader._iterator._reset(dataloader)
#             raise StopIteration
#         else:
#             if self.n_domains_per_batch <= len(nonzero_indices):
#                 groups_for_batch = np.random.choice(nonzero_indices, size=self.n_domains_per_batch, replace=False)
#             else:
#                 groups_for_batch = np.random.choice(nonzero_indices, size=len(nonzero_indices), replace=False)
#             batches = []
#             for batch_num in groups_for_batch:
#                 self.n_per_domain[batch_num] -= 1
#                 # TODO: whether next(iter(?))
#                 batch = next(iter(self.dataloaders[batch_num]))
#                 # print(batch_num, torch.sum(batch[0]).item())
#                 # print("------")
#                 batches.append(batch)
#             return batches


## Copied from https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):

    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, init_lr=1e-7, after_scheduler=None):
        self.init_lr = init_lr
        assert init_lr > 0, 'Initial LR should be greater than 0.'
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if (self.finished and self.after_scheduler) or self.total_epoch == 0:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

