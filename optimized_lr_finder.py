import sys
sys.path.insert(0, 'pytorch-lr-finder/torch_lr_finder/')
from config import *
import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import threading
import curses 
import gc
import time
from random import choices
from itertools import chain
import numpy as np
import pandas as pd
import sklearn
import cv2
from tqdm import tqdm as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from apex import amp
import torch, torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from MelanomaDataset import MelanomaDataset
from catalyst.data.sampler import BalanceClassSampler
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from utils import *
from optimizers import Over9000
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
from config import *
from torch_lr_finder import LRFinder
prev_epoch_num = 0
best_valid_loss = np.inf
best_valid_auc = 0.0
balanced_sampler = False
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
pseduo_df = pseudo_label_df(pseduo_df, pseudo_lo_thr, pseudo_up_thr)
pseudo_labels = list(pseduo_df['target'])
print("Pseudo data length: {}".format(len(pseduo_df)))
print("Negative label: {}, Positive label: {}".format(pseudo_labels.count(0), pseudo_labels.count(1))) 
df = pd.read_csv('data/folds.csv')
pseduo_df['fold'] = np.nan
pseduo_df['fold'] = pseduo_df['fold'].map(lambda x: n_fold)
X, y = df['image_id'], df['target']
# train_df['fold'] = np.nan
df = meta_df(df, image_path)
pseduo_df = meta_df(pseduo_df, test_image_path)

#split data
# mskf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
# for i, (_, test_index) in enumerate(mskf.split(X, y)):
#     train_df.iloc[test_index, -1] = i
    
df['fold'] = df['fold'].astype('int')
idxs = [i for i in range(len(df))]
train_idx = []
val_idx = []
train_df = df[df['fold'] != fold]
train_df = pd.concat([train_df, pseduo_df], ignore_index=True)
valid_df = df[(df['fold'] == fold) & (df['source'] == 'ISIC20')]
test_df = pseduo_df
train_meta = np.array(train_df[meta_features].values, dtype=np.float32)
valid_meta = np.array(valid_df[meta_features].values, dtype=np.float32)
test_meta = np.array(test_df[meta_features].values, dtype=np.float32)
# model = seresnext(pretrained_model).to(device)
model = EffNet(pretrained_model=pretrained_model, n_meta_features=train_meta.shape[1], freeze_upto=freeze_upto).to(device)

train_ds = MelanomaDataset(train_df.path.values, train_meta, train_df.target.values, dim=sz, transforms=train_aug)
if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=BalanceClassSampler(labels=train_ds.get_labels(), mode="downsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, num_workers=4)

plist = [
        {'params': model.backbone.parameters(),  'lr': learning_rate/50},
        {'params': model.meta_fc.parameters(),  'lr': learning_rate},
        # {'params': model.metric_classify.parameters(),  'lr': learning_rate},
    ]

optimizer = optim.Adam(plist, lr=learning_rate)
criterion = criterion_margin_focal_binary_cross_entropy

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=100, num_iter=1000)
lr_finder.plot() # to inspect the loss-learning rate graph
