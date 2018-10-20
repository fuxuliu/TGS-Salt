import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

from contextlib import contextmanager
import datetime
import  time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from losses import Loss
from losses import Scheduler
from losses.Evaluation import do_kaggle_metric, dice_accuracy, do_mAP, unpad_im


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.3f}s".format(title, time.time() - t0))


## A class for model(train, valid, and predict)
class Seg(nn.Module):

    def __init__(self, lr=0.005, fold=None, val_mode='max'):
        super(Seg, self).__init__()
        self.lr = lr
        self.fold = fold
        self.scheduler = None
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        self.create_save_folder()

    ## define the optimizer and scheduler
    def optmizer_and_scheduler(self, T_max=10, T_mul=2, lr_min=0):
        self.cuda()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.lr, momentum=0.9, weight_decay=0.0001)
        self.scheduler = Scheduler.CosineAnnealingLR(self.optimizer,
                                                        T_max=T_max,
                                                        T_mul=T_mul,
                                                        lr_min=lr_min,
                                                        val_mode=self.val_mode,
                                                        last_epoch=-1,
                                                        save_snapshots=True)


    def train_and_valid(self, train_loader, val_loader, n_epoch=10):
        print('Model created, total of {} parameters'.format(
            sum(p.numel() for p in self.parameters())))
        while self.epoch < n_epoch:
            self.epoch += 1
            lr = np.mean([param_group['lr'] for param_group in self.optimizer.param_groups])
            with timer('Train Epoch {:}/{:} - LR: {:.3E}'.format(self.epoch, n_epoch, lr)):
                # Training step
                train_loss, train_closs, train_acc, train_nonem_loss, train_iou, train_mAP = self.do_train(train_loader)
                #  Validation
                val_loss, val_closs, val_acc, val_nonem_loss, val_iou, val_mAP = self.do_valid(val_loader)
                # Learning Rate Scheduler
                self.scheduler.step(self.epoch,
                                    save_dict=dict(metric=np.mean(val_mAP),
                                                    save_dir=self.save_dir,
                                                    fold=self.fold,
                                                    state_dict=self.state_dict()))
            # Print statistics
            print(('train loss: {:.3f}  val_loss: {:.3f}  '
                    'train closs: {:.3f}  val_closs: {:.3f}  '
                    'train acc: {:.3f}  val_acc: {:.3f}  '
                    'train non_loss: {:.3f}  val_non_loss: {:.3f}  '
                   'train iou:  {:.3f}  val_iou:  {:.3f}  '
                   'train mAP:  {:.3f}  val_mAP:  {:.3f}').format(
                np.mean(train_loss),
                np.mean(val_loss),
                np.mean(train_closs),
                np.mean(val_closs),
                np.mean(train_acc),
                np.mean(val_acc),
                np.mean(train_nonem_loss),
                np.mean(val_nonem_loss),
                np.mean(train_iou),
                np.mean(val_iou),
                np.mean(train_mAP),
                np.mean(val_mAP)))

    ## training the model one training set
    def do_train(self, train_loader):
        self.set_mode('train')
        train_loss = []
        train_closs = []
        train_acc = []
        train_nonem_loss = []
        train_iou = []
        train_mAP = []
        for i, (index, im, mask, targets) in enumerate(train_loader):
            non_empty_index = [i for i in range(len(targets)) if targets[i] == 1]
            self.optimizer.zero_grad()
            im = im.cuda()
            mask = mask.cuda()
            targets = targets.reshape((-1)).cuda()
            logit, logit_pixel, logit_image = self.forward(im)
            
            loss_image = F.binary_cross_entropy_with_logits(logit_image, targets, reduce=True)

            out = logit_image.ge(0.5).float()
            correct = (targets == out).sum()
            a = correct.item() / targets.size(0)
            pred = torch.sigmoid(logit)

            loss = self.criterion(logit, mask)
            non_empty_loss = self.criterion(logit_pixel[non_empty_index], mask[non_empty_index])
            iou  = dice_accuracy(pred, mask, is_average=False)
            mAP = do_mAP(pred.data.cpu().numpy(), mask.cpu().numpy(), is_average=False)

            train_loss.append(loss.item())
            train_closs.append(loss_image.item())
            train_acc.append(a)
            if non_empty_loss != 0:
                train_nonem_loss.append(non_empty_loss.item())
            train_iou.extend(iou)
            train_mAP.extend(mAP)

            sum_loss = 1.0*loss + 0.5*non_empty_loss+ 0.05*loss_image
            # sum_loss = 1.0*loss + 0.10*non_empty_loss+ 0.05*loss_image
            sum_loss.backward()
            self.optimizer.step()
        return train_loss, train_closs, train_acc, train_nonem_loss, train_iou, train_mAP


    # evaluate the socre on val_dataset
    def do_valid(self, val_loader):
        self.set_mode('valid')
        val_loss = []
        val_closs = []
        val_acc = []
        val_nonem_loss = []
        val_iou = []
        val_mAP = []
        for index, im, mask, targets in val_loader:
            non_empty_index = [i for i in range(len(targets)) if targets[i] == 1]
            im = im.cuda()
            mask = mask.cuda()
            targets = targets.reshape((-1)).cuda()

            with torch.no_grad():
                logit, logit_pixel, logit_image = self.forward(im)
                pred = torch.sigmoid(logit)
                loss = self.criterion(logit, mask)
                non_empty_loss = self.criterion(logit_pixel[non_empty_index], mask[non_empty_index])
                loss_image = F.binary_cross_entropy_with_logits(logit_image, targets, reduce=True)
                out = logit_image.ge(0.5).float()
                correct = (targets == out).sum()
                acc = correct.item() / targets.size(0)
                iou  = dice_accuracy(pred, mask, is_average=False)
                mAP = do_mAP(pred.cpu().numpy(), mask.cpu().numpy(), is_average=False)

            val_loss.append(loss.item())
            val_closs.append(loss_image.item())
            val_acc.append(acc)
            if (non_empty_loss != 0):
                val_nonem_loss.append(non_empty_loss.item())
            val_iou.extend(iou)
            val_mAP.extend(mAP)

        return val_loss, val_closs, val_acc, val_nonem_loss, val_iou, val_mAP

    # predict the test set
    def do_predict(self, test_loader, tta_transform=None, threshold=0.45):
        self.set_mode('test')
        self.cuda()
        for i, (idx, im) in enumerate(test_loader):
            with torch.no_grad():
                # Apply TTA and predict
                batch_pred = []
                prob_pred = []
                # TTA
                if tta_transform is not None:
                    tta_list = torch.FloatTensor(tta_transform(im.cpu().numpy(), mode='in'))
                    tta_pred = []
                    for t_im in tta_list:
                        t_im = t_im.cuda()
                        t_logit, logit_pixel, logit_image = self.forward(t_im)
                        pred = torch.sigmoid(t_logit)
                        pred = unpad_im(pred.cpu().numpy())
                        tta_pred.append(pred)
                    batch_pred.extend(tta_transform(tta_pred, mode='out'))

                # Predict original batch
                im = im.cuda()
                logit, logit_pixel, logit_image = self.forward(im)
                pred = torch.sigmoid(logit)
                pred = unpad_im(pred.cpu().numpy())
                batch_pred.append(pred)

                # Average TTA results
                batch_pred = np.mean(batch_pred, 0)
                # Threshold result
                if threshold > 0:
                    batch_pred = batch_pred > threshold

                if not i:
                    out = batch_pred
                    ids = idx

                else:
                    out = np.concatenate([out, batch_pred], axis=0)
                    ids = np.concatenate([ids, idx], axis=0)
        out = dict(id=ids, pred=out)
        return out

    # define the loss function
    def define_criterion(self, name):
        if name.lower() == 'bce+dice':
            self.criterion = Loss.BCE_Dice()
        elif name.lower() == 'dice':
            self.criterion = Loss.DiceLoss()
        elif name.lower() == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif name.lower() == 'robustfocal':
            self.criterion = Loss.RobustFocalLoss2d()
        elif name.lower() == 'lovasz-hinge' or name.lower() == 'lovasz':
            self.criterion = Loss.Lovasz_Hinge(per_image=True)
        elif name.lower() == 'bce+lovasz':
            self.criterion = Loss.BCE_Lovasz(per_image=True)
        else:
            raise NotImplementedError('Loss {} is not implemented'.format(name))

    ## set the mode for model(train or eval)
    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    # loading the model
    def load_model(self, path=None):
        self.load_state_dict(torch.load(path))

    def create_save_folder(self):
        name = type(self).__name__
        self.save_dir = os.path.join('./Saves', name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
