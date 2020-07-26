import os
import numpy as np
import cv2
import pandas as pd 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F_alb

gen_challenge = {'lower extremity': 2, 'torso':3, 'head/neck':0, 'oral/genital':5, 'palms/soles':4, 'nan':-1, 'upper extremity':1}

def pseudo_label_df(df, lo_th=0.1, up_th=0.8):
    pred = df['prediction'].copy()
    pred[pred<lo_th] = 0
    pred[pred>up_th] = 1
    df['prediction'] = pred
    df = df.drop(df[(df.prediction> 0) & (df.prediction < 1)].index)
    df['target'] = df['prediction'].astype('int')
    return df

def rank_based_pseudo_label_df(df, image_path, top=3000, bottom=150):
    df['prediction'] = df['prediction'].astype('float')
    # df.sort_values(by=['prediction'], inplace=True)
    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].map(gen_challenge)
    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna(-1)
    # Sex features
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['sex'] = df['sex'].fillna(-1)
    # Age features
    # df['age_approx'] /= df['age_approx'].max()
    df['age_approx'] = df['age_approx'].fillna(0)
    df['patient_id'] = df['patient_id'].fillna(0)
    top_val = df['prediction'][top]
    bottom_val = df['prediction'][len(df)-bottom]
    
    df = df.drop(df[(df.prediction> top_val) & (df.prediction < bottom_val)].index)
    pred = df['prediction'].copy()
    pred[pred<=top_val] = 0
    pred[pred>=bottom_val] = 1
    df['prediction'] = pred
    df['target'] = df['prediction'].astype('int')

    try:
        df['image_name'] = df['image_id'].map(lambda x: os.path.join(image_path,'{}.jpg'.format(x)))
    except:
        df['image_name'] = df['image_name'].map(lambda x: os.path.join(image_path,'{}.jpg'.format(x)))
    return df

def meta_df(df, image_path):
    '''
    Meta features: https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
    '''
    # One-hot encoding of anatom_site_general_challenge feature
    concat = df['anatom_site_general_challenge']
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)

    # Sex features
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['sex'] = df['sex'].fillna(-1)

    # Age features
    # df['age_approx'] /= df['age_approx'].max()
    df['age_approx'] = df['age_approx'].fillna(0)
    df['patient_id'] = df['patient_id'].fillna(0)
    try:
        df['path'] = df['image_id'].map(lambda x: os.path.join(image_path,'{}.jpg'.format(x)))
    except:
        df['path'] = df['image_name'].map(lambda x: os.path.join(image_path,'{}.jpg'.format(x)))

    return df

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
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
    
def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets, shuffled_targets, lam]
    return data, targets

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets, shuffled_targets, lam]
    return data, targets

def cutmix_criterion(preds, targets, criterion, rate=0.7):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    return lam * ohem_loss(rate, criterion, preds, targets1) + (1 - lam) * ohem_loss(rate, criterion, preds, targets2)
    # return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def mixup_criterion(preds, targets, criterion, rate=0.7):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    return lam * ohem_loss(rate, criterion, preds, targets1) + (1 - lam) * ohem_loss(rate, criterion, preds, targets2)
    # return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view).cuda()
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)

def ohem_loss(rate, base_crit, cls_pred, cls_target):

    batch_size = cls_pred.size(0) 
    # ohem_cls_loss = base_crit(cls_pred, cls_target, reduction='none', ignore_index=-1)
    ohem_cls_loss = base_crit(cls_pred, cls_target)
    if rate==1:
        return ohem_cls_loss.sum()
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min((sorted_ohem_loss.size())[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss

def save_model(valid_loss, valid_auc, best_valid_loss, best_valid_auc, best_state, savepath):
    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}. Saving checkpoint')
        torch.save(best_state, savepath+'_loss.pth')
        best_valid_loss = valid_loss
    if valid_auc>best_valid_auc:
        print(f'Validation auc has increased from:  {best_valid_auc:.4f} to: {valid_auc:.4f}. Saving checkpoint')
        torch.save(best_state, savepath + '_auc.pth')
        best_valid_auc = valid_auc
    return best_valid_loss, best_valid_auc 