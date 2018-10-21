# TGS-Salt
56th place(top2%) solution for Kaggle TGS Salt Identification Challenge

## General

This is a not bad solution to get top2% place without post-processing.  
Accodring to the forum, [binary empty vs non-empty classifier](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933#latest-406444) by [Heng](https://www.kaggle.com/hengck23) and [+0.01 LB with snapshot ensembling and cyclic lr](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65347) by [Peter](https://www.kaggle.com/pestipeti) and so on,There are many useful tricks.

## My solution
#### Augmentation:
I used padding image from 101x101 to 128*128, but I did not compared it with just resize. Some said the resize+flip is better than pad+aug.
you can check the code here[transform.py](https://github.com/Gary-Deeplearning/TGS-Salt/blob/master/dataset/transform.py)

#### pretrained model
I used the resnet34 pretrained model and se-resnext50 pretraied model and se-resnext101 as the Unet encoder. From the results of experiments, the se-resnext50 pretrained model is the best Unet encoder, but some top kagglers said their best model is resnet34.

#### scSE and hypercolumn
I used the [scSE block](https://arxiv.org/pdf/1803.02579) and [hypercolumn](https://arxiv.org/abs/1411.5752) on decoder. It can raise the score a little bit.

#### deep supervision
[binary empty vs non-empty classifier](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933#latest-406444). Deep supervision can help the model converge quickly and increase the LB score.

#### Loss function
From the results of experiments, train with only [lovasz_loss and elu+1](https://github.com/bermanmaxim/LovaszSoftmax) is better than train model with bce in stage#1 and lovasz in stage#2.

#### LR_Scheduler
SGDR with cycle learing rate.

## Other excellent Solutions
[5th place by AlexenderLiao](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69051)<br>
[9th place by tugstugi](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053#latest-406939)<br>
[5th place by Alecander Liao](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69051)<br>
[11th place by alexisrozhkov](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69093)
[22nd place by Vishunu](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69101#latest-407311)<br>
[43th place by n01z3](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69039)<br>

