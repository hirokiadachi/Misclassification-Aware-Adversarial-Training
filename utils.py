import os
import sys
import time
import math
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

##########################################################################
# Impremented softmax crossentropy loss for soft label
##########################################################################
class CustomLossFunction:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        
    def kldiv_loss(self, p, q):
        b, c = p.shape
        p = torch.softmax(p, dim=1)
        if self.reduction == 'mean':
            loss = torch.sum(p * torch.log(p/q)) / b
        elif self.reduction == 'sum':
            loss = torch.sum(p * torch.log(p/q))
        elif self.reduction == 'none':
            loss = torch.sum(p * torch.log(p/q), keepdim=True)
        return loss
    
    def softlabel_ce(self, x, t):
        b, c = x.shape
        x_log_softmax = torch.log_softmax(x, dim=1)
        if self.reduction == 'mean':
            loss = -torch.sum(t*x_log_softmax) / b
        elif self.reduction == 'sum':
            loss = -torch.sum(t*x_log_softmax)
        elif self.reduction == 'none':
            loss = -torch.sum(t*x_log_softmax, keepdims=True)
        return loss
    
    def cw_loss(self, x, t):
        correct = torch.sum(t * x, dim=1)
        wrong = torch.max((1 - t) * x - 1e4*t, dim=1)[0]
        loss = -F.relu(correct - wrong + 50)
        return loss.mean()

##########################################################################
# the function to calclate accuracy
##########################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

##########################################################################
# the function to derive masking regions
##########################################################################
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def normalize(x, mu, std):
    return (x - mu) / std
            

##########################################################################
def prob_generator(onehot, num_classes, device, num_positive=None):
    ys1 = torch.zeros_like(onehot).to(device)
    neg_idex = torch.where((1 - onehot)==1)[1].view(-1, num_classes - 1).data.cpu().numpy()
    gt_index = onehot.argmax(dim=1).data.cpu().numpy()
    num_positive = max(min(num_positive, num_classes - 2), 10) ## probably, minimum of positive index should use an arbitrary the number of classes (e.g. 10 classes), not 1.
    
    for y1, index in zip(ys1, neg_idex):
        y1[random.sample(index.tolist(), num_positive)] = 1
    
    ys2 = (1 - ys1) * (1 - onehot)
    ys1 *= (1 / num_positive)
    ys2 *= (1 / (num_classes - num_positive - 1))
    return ys1, ys2
    

def save_checkpoint(state, is_best_clean, is_best_adv, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    print('Model save..')
    torch.save(state, filepath)
    if is_best_clean and is_best_adv:
        print('==> Updating the best model..')
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))