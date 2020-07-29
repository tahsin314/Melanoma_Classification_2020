import os
os.environ['XLA_USE_BF16'] = "1"
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
from MelanomaDataset import MelanomaDataset, MelanomaDataLoader
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from utils import *
from optimizers import Over9000
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
from config import *

balanced_sampler = False
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
df = pd.read_csv('data/train_768.csv')
    
df['fold'] = df['fold'].astype('int')
idxs = [i for i in range(len(df))]
train_idx = []
val_idx = []
train_folds = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15]
valid_folds = [4, 9, 14]
train_df = df[df['fold'] == train_folds[0]]
valid_df = df[df['fold'] == valid_folds[0]]

for i in train_folds[1:]:
  train_df = pd.concat([train_df, df[df['fold'] == i]])
for i in valid_folds[1:]:
  valid_df = pd.concat([valid_df, df[df['fold'] == i]])
# test_df = pseduo_df
train_meta = np.array(train_df[meta_features].values, dtype=np.float32)
valid_meta = np.array(valid_df[meta_features].values, dtype=np.float32)
# test_meta = np.array(test_df[meta_features].values, dtype=np.float32)
# model = seresnext(pretrained_model).to(device)

def Engine_Tahsin():
  device = xm.xla_device()
  model = EffNet(pretrained_model=pretrained_model, freeze_upto=freeze_upto)
  model = model.to(device)
  train_loader = MelanomaDataLoader(train_df.image_name.values, train_meta, train_df.target.values, sz, train_aug).fetch(
          batch_size=batch_size, 
          drop_last=True, 
          num_workers=0, 
          shuffle=True, 
          tpu=True
      )

  valid_loader = MelanomaDataLoader(valid_df.image_name.values, valid_meta, valid_df.target.values, sz, train_aug).fetch(
          batch_size=batch_size, 
          drop_last=True, 
          num_workers=0, 
          shuffle=True, 
          tpu=True
      )

  # Effnet model
  plist = [ 
          {'params': model.backbone.parameters(),  'lr': learning_rate/50},
          {'params': model.meta_fc.parameters(),  'lr': learning_rate},
          {'params': model.output.parameters(),  'lr': learning_rate},
      ]

  optimizer = optim.Adam(plist, lr=learning_rate)
  lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
  cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/10, max_lr=learning_rate, step_size_up=2*len(train_loader), step_size_down=2*len(train_loader), mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

  # nn.BCEWithLogitsLoss(), ArcFaceLoss(), FocalLoss(logits=True).to(device), LabelSmoothing().to(device) 
  criterion = criterion_margin_focal_binary_cross_entropy

  if load_model:
    tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
    model.load_state_dict(tmp['model'])
    # optimizer.load_state_dict(tmp['optim'])
    # lr_reduce_scheduler.load_state_dict(tmp['scheduler'])
    # cyclic_scheduler.load_state_dict(tmp['cyclic_scheduler'])
    # amp.load_state_dict(tmp['amp'])
    prev_epoch_num = tmp['epoch']
    best_valid_loss = tmp['best_loss']
    best_valid_loss, best_valid_auc = train_val(-1, valid_loader, optimizer=optimizer, device=device, rate=1, train=False, mode='val')
    del tmp
    print('Model Loaded!')

  prev_epoch_num = 0
  best_valid_loss = np.inf
  best_valid_auc = 0.0  
  for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())
    model = train_val(epoch, train_loader, optimizer=optimizer, device=device, model=model, criterion=criterion, choice_weights=choice_weights, rate=1.00, train=True, mode='train')
    valid_loss, valid_auc = train_val(epoch, valid_loader, optimizer=optimizer, device=device, model=model, criterion=criterion, rate=1.00, train=False, mode='val')
    print("#"*20)
    print(f"Epoch {epoch} Report:")
    print(f"Validation Loss: {valid_loss :.4f} \n Validation AUC: {valid_auc :.4f}")
    best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler':lr_reduce_scheduler.state_dict(), 'cyclic_scheduler':cyclic_scheduler.state_dict(), 
          # 'amp': amp.state_dict(),
    'best_loss':valid_loss, 'best_auc':valid_auc, 'epoch':epoch}
    best_valid_loss, best_valid_auc = save_model(valid_loss, valid_auc, best_valid_loss, best_valid_auc, best_state, os.path.join(model_dir, model_name))
    print("#"*20)

def train_val(epoch, dataloader, optimizer, device, model, criterion, choice_weights= [0.8, 0.1, 0.1], rate=1, train=True, mode='train'):
  t1 = time.time()
  running_loss = 0
  epoch_samples = 0
  pred = []
  lab = []
  if train:
    model.train()
    print("Initiating train phase ...")
  else:
    model.eval()
    print("Initiating val phase ...")
  para_loader = pl.ParallelLoader(dataloader, [device])
  tk0 = T(para_loader.per_device_loader(device), total=len(dataloader), disable=xm.get_ordinal()==0)
  for idx, (_, inputs,meta,labels) in enumerate(tk0):
    with torch.set_grad_enabled(train):
      inputs = inputs.to(device)
      meta = meta.to(device)
      labels = labels.to(device)
      epoch_samples += len(inputs)
      if not train:
        choice_weights = [1.0, 0, 0]
      choice = choices(opts, weights=choice_weights)
      optimizer.zero_grad()
      # with torch.cuda.amp.autocast(mixed_precision):
      if choice[0] == 'normal':
        outputs = model(inputs.float(), meta)
        loss = ohem_loss(rate, criterion, outputs, labels)
        reduced_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
        running_loss += reduced_loss.item()
      
      elif choice[0] == 'mixup':
        inputs, targets = mixup(inputs, labels, np.random.uniform(0.8, 1.0))
        outputs = model(inputs.float(), meta)
        loss = mixup_criterion(outputs, targets, criterion=criterion, rate=rate)
        reduced_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
        running_loss += reduced_loss.item()
      
      elif choice[0] == 'cutmix':
        inputs, targets = cutmix(inputs, labels, np.random.uniform(0.8, 1.0))
        outputs = model(inputs.float(), meta)
        loss = cutmix_criterion(outputs, targets, criterion=criterion, rate=rate)
        reduced_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
        running_loss += reduced_loss.item()
      if train:
        if mixed_precision:
          scaler.scale(loss).backward()
          if (idx+1) % accum_step == 0:
            scaler.step(optimizer) 
            scaler.update() 
        else:
          loss.backward()
          if (idx+1) % accum_step == 0:
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            # cyclic_scheduler.step()    
      elapsed = int(time.time() - t1)
      eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
      pred.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
      lab.extend(torch.argmax(labels, 1).cpu().numpy())
      if train:
        msg = f"Epoch: {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s"
      else:
        msg = f'Epoch {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
      # xm.master_print(msg)
      tk0.set_postfix_str(msg)
  history.loc[epoch, f'{mode}_loss'] = running_loss/epoch_samples
  history.loc[epoch, f'{mode}_time'] = elapsed  
  if mode=='val':
    lr_reduce_scheduler.step(running_loss)
    auc = roc_auc_score(lab, pred)
    msg = f'{mode} Loss: {running_loss/epoch_samples:.4f} \n {mode} Auc: {auc:.4f}'
    xm.master_print(msg)
    history.loc[epoch, f'{mode}_loss'] = running_loss/epoch_samples
    history.loc[epoch, f'{mode}_auc'] = auc
    history.to_csv(f'history_{model_name}.csv', index=False)
    return running_loss/epoch_samples, auc
  else:
    return model 
def main(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = Engine_Tahsin()

if __name__ == '__main__':
  FLAGS={}
  xmp.spawn(main, args=(FLAGS,), nprocs=8, start_method='fork')