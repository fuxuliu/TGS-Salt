import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovasz_loss import lovasz_hinge, binary_xloss
################### DICE ########################
def IoU(logit, truth, smooth=1):
    prob = torch.sigmoid(logit)
    intersection = torch.sum(prob * truth)
    union = torch.sum(prob + truth)
    iou = (2 * intersection + smooth) / (union + smooth)
    return iou

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logit, truth):
        iou = IoU(logit, truth, self.smooth)
        loss = 1 - iou
        return loss

################ FOCAL LOSS ####################
class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers

    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

################# BCE + DICE ########################
class BCE_Dice(nn.Module):
    def __init__(self, smooth=1):
        super(BCE_Dice, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit, truth):
        dice = self.dice(logit, truth)
        bce = self.bce(logit, truth)
        return dice + bce

############### LOVÁSZ-HINGE ########################
class Lovasz_Hinge(nn.Module):
    def __init__(self, per_image=True):
        super(Lovasz_Hinge, self).__init__()
        self.per_image = per_image

    def forward(self, logit, truth):
        return lovasz_hinge(logit, truth,
                            per_image=self.per_image)


############## BCE + LOVÁSZ #########################
class BCE_Lovasz(nn.Module):
    def __init__(self, per_image=True):
        super(BCE_Lovasz, self).__init__()
        self.per_image = per_image

    def forward(self, logit, truth):
        bce = binary_xloss(logit, truth)
        lovasz = lovasz_hinge(logit, truth, per_image=self.per_image)
        return bce + lovasz