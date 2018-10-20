import datetime
import math
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implementation
    contains restarts and T_mul.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mul, lr_min=0, last_epoch=-1, val_mode='max', save_snapshots=False):
        self.T_max = T_max
        self.T_mul = T_mul
        self.T_curr = 0
        self.lr_min = lr_min
        self.save_snapshots = save_snapshots
        self.val_mode = val_mode
        self.best_model_path = None
        self.reset = 0

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.T_curr % self.T_max

        if not r and self.last_epoch > 0:
            self.T_max *= self.T_mul
            self.T_curr = 1
            self.update_saving_vars()
        else:
            self.T_curr += 1

        return [self.lr_min + (base_lr - self.lr_min) *
                (1 + math.cos(math.pi * r / self.T_max)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None, save_dict=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.save_snapshots and save_dict is not None:
            self.save_best_model(save_dict)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def update_saving_vars(self):
        self.reset += 1
        self.best_model_path = None

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf


    def save_best_model(self, save_dict):
        metric = save_dict['metric']
        fold = save_dict['fold']
        save_dir = save_dict['save_dir']
        state_dict = save_dict['state_dict']

        if (self.val_mode == 'max' and metric > self.best_metric) or (
                self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            if fold is not None:
                self.best_model_path = os.path.join(
                    save_dir,
                    '{:}_Fold{:}_Epoach{}_reset{:}_val{:.3f}'.format(date, fold, self.last_epoch, self.reset, metric))
            else:
                self.best_model_path = os.path.join(
                    save_dir,
                    '{:}_Epoach{}_reset{:}_val{:.3f}'.format(date, self.last_epoch, self.reset, metric))

            torch.save(state_dict, self.best_model_path)
