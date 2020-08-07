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
import torch, torchvision
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

if mixed_precision:
  scaler = torch.cuda.amp.GradScaler()

prev_epoch_num = 0
best_valid_loss = np.inf
best_valid_auc = 0.0
balanced_sampler = False
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
pseduo_df = rank_based_pseudo_label_df(pseduo_df, 0.2, 0.99)
pseudo_labels = list(pseduo_df['target'])
print("Pseudo data length: {}".format(len(pseduo_df)))
print("Negative label: {}, Positive label: {}".format(pseudo_labels.count(0), pseudo_labels.count(1))) 
df = pd.read_csv('data/train_768.csv')
pseduo_df['fold'] = np.nan
pseduo_df['fold'] = pseduo_df['fold'].map(lambda x: 16)
# pseduo_df = meta_df(pseduo_df, test_image_path)
    
df['fold'] = df['fold'].astype('int')
idxs = [i for i in range(len(df))]
train_idx = []
val_idx = []
train_folds = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16]
valid_folds = [4, 9, 14]
train_df = df[df['fold'] == train_folds[0]]
valid_df = df[df['fold'] == valid_folds[0]]

for i in valid_folds[1:]:
  valid_df = pd.concat([valid_df, df[df['fold'] == i]])
valid_meta = np.array(valid_df[meta_features].values, dtype=np.float32)
model = seresnext(pretrained_model, use_meta=True).to(device)
# model = EffNet(pretrained_model=pretrained_model, freeze_upto=freeze_upto).to(device)

valid_ds = MelanomaDataset(valid_df.image_name.values, valid_meta, valid_df.target.values, dim=sz, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

def train_val(epoch, dataloader, optimizer, choice_weights= [0.8, 0.1, 0.1], rate=1):
  t1 = time.time()
  running_loss = 0
  epoch_samples = 0
  img_ids = []
  pred = []
  lab = []
  
  model.eval()
  print("Initiating val phase ...")
  for idx, (img_id, inputs,meta,labels) in enumerate(dataloader):
    with torch.set_grad_enabled(False):
        inputs = inputs.to(device)
        meta = meta.to(device)
        labels = labels.to(device)
        epoch_samples += len(inputs)
        choice_weights = [1.0, 0, 0]
        choice = choices(opts, weights=choice_weights)
        optimizer.zero_grad()
        outputs = model(inputs.float(), meta)
        loss = ohem_loss(1.00, criterion, outputs, labels)
        running_loss += loss.item() 
        elapsed = int(time.time() - t1)
        eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
        img_ids.extend(img_id)
        pred.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
        lab.extend(torch.argmax(labels, 1).cpu().numpy())
        msg = f'Epoch {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
        print(msg, end= '\r')
  
  score_diff = np.abs(np.array(pred)-np.array(lab))      
  auc = roc_auc_score(lab, pred)
  zippedList =  list(zip(img_ids, lab, pred, score_diff))
  submission = pd.DataFrame(zippedList, columns = ['image_name','label', 'target', 'difference'])
  submission = submission.sort_values(by=['difference'], ascending=False)
  submission.to_csv('val_report.csv', index=False)
  msg = f'Validation Loss: {running_loss/epoch_samples:.4f} \n Validation Auc: {auc:.4f}'
  print(msg)
  return running_loss/epoch_samples, auc

# Effnet model
plist = [ 
        {'params': model.backbone.parameters(),  'lr': learning_rate/50},
        # {'params': model.meta_fc.parameters(),  'lr': learning_rate},
        # {'params': model.output.parameters(),  'lr': learning_rate},
    ]

optimizer = optim.Adam(plist, lr=learning_rate)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

# nn.BCEWithLogitsLoss(), ArcFaceLoss(), FocalLoss(logits=True).to(device), LabelSmoothing().to(device) 
criterion = criterion_margin_focal_binary_cross_entropy

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  if mixed_precision:
    scaler.load_state_dict(tmp['scaler'])
  # amp.load_state_dict(tmp['amp'])
  prev_epoch_num = tmp['epoch']
  best_valid_loss = tmp['best_loss']
  print('Model Loaded!')

valid_loss, valid_auc = train_val(-1, valid_loader, optimizer=optimizer, rate=1.00)