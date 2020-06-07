import os
import shutil
import sys
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
from utils import *
from metrics import *
from optimizers import Over9000
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from model.seresnext import seresnext
# from model.effnet import EfficientNetWrapper
# from model.densenet import *
## This library is for augmentations .
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    Resize,
    CenterCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    Cutout,
    RandomGamma,
    ShiftScaleRotate ,
    GaussNoise,
    Blur,
    MotionBlur,   
    GaussianBlur,
    Normalize, 
)
n_fold = 5
fold = 1
SEED = 24
batch_size = 48
sz = 256
learning_rate = 1e-3
patience = 5
opts = ['normal', 'mixup', 'cutmix']
device = 'cuda:0'
# device = 'cpu:0'
apex = False
pretrained_model = 'seresnext50_32x4d'
model_name = '{}_trial_stage1_fold_{}'.format(pretrained_model, fold)
model_dir = 'model_dir'
history_dir = 'history_dir'
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
load_model = False
history = pd.DataFrame()
prev_epoch_num = 0
n_epochs = 3
valid_recall = 0.0
best_valid_recall = 0.0
best_valid_loss = np.inf
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)


train_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    # GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    # RandomAugMix(severity=1, width=1, alpha=1., p=0.3),
    # OneOf([
    #     ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=30,border_mode=cv2.BORDER_CONSTANT,value =0),
    #     GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =0, p=0.1),
    #     OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =0)                  
    #     ], p=0.3),
    OneOf([
        GaussNoise(var_limit=0.02),
        # Blur(),
        GaussianBlur(blur_limit=3),
        RandomGamma(p=0.8),
        ], p=0.5),
    Normalize()
    ]
      )
val_aug = Compose([Normalize()])
train_df = pd.read_csv('data/train.csv')
X, y = train_df['image_name'], train_df['target']
train_df['fold'] = np.nan
train_df= train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
#split data
mskf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    train_df.iloc[test_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')
idxs = [i for i in range(len(train_df))]
train_idx = []
val_idx = []
model = seresnext(pretrained_model).to(device)

# For stratified split
for i in T(range(len(train_df))):
    if train_df.iloc[i]['fold'] == fold: val_idx.append(i)
    else: train_idx.append(i)

# train_idx = idxs[:int((n_fold-1)*len(idxs)/(n_fold))]
# train_idx = np.load('train_pseudo_idxs.npy')
# val_idx = idxs[int((n_fold-1)*len(idxs)/(n_fold)):]

train_ds = MelanomaDataset('data/train.csv', train_idx[:320], transforms=train_aug)
train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True)

valid_ds = MelanomaDataset('data/train.csv', val_idx[:320], transforms=None)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

## This function for train is copied from @hanjoonchoe
## We are going to train and track accuracy and then evaluate and track validation accuracy

def train(epoch,history):
  t1 = time.time()
  model.train()
  losses = []
  accs = []
  acc= 0.0
  total = 0.0
  running_loss = 0.0
  rate = 1
  
  if epoch<10:
    rate = 1
  elif epoch>=10 and rate>0.65:
    rate = np.exp(-(epoch-30)/60)
  else:
    rate = 0.65
  for idx, (inputs,labels) in enumerate(train_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    total += len(inputs)
    choice = choices(opts, weights=[0.70, 0.15, 0.15])
    optimizer.zero_grad()
    if choice[0] == 'normal':
      outputs = model(inputs.float())
      loss = criterion(outputs,labels)
      running_loss += loss.item()
    
    elif choice[0] == 'mixup':
      inputs, targets = mixup(inputs, labels, np.random.uniform(0.8, 1.0))
      outputs = model(inputs.float())
      loss = mixup_criterion(outputs, targets, rate=rate)
      running_loss += loss.item()
    
    elif choice[0] == 'cutmix':
      inputs, targets = cutmix(inputs, labels, np.random.uniform(0.8, 1.0))
      outputs = model(inputs.float())
      loss = cutmix_criterion(outputs, targets, rate=rate)
      running_loss += loss.item()
    if apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
          loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()
    elapsed = int(time.time() - t1)
    eta = int(elapsed / (idx+1) * (len(train_loader)-(idx+1)))
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if idx%1==0:
      msg = 'Epoch: {} \t Progress: {}/{} \t Loss: {:.4f} \t Time: {}s \t ETA: {}s'.format(epoch, 
      idx, len(train_loader), running_loss/(idx+1), elapsed, eta)
      print(msg, end='\r')

  losses.append(running_loss/len(train_loader))
  
  torch.cuda.empty_cache()
  gc.collect()
  history.loc[epoch, 'train_loss'] = losses[0]
  history.loc[epoch, 'Time'] = elapsed

def evaluate(epoch,history):
   model.eval()
   total = 0.0
   running_loss = 0.0
   pred = []
   lab = []
   with torch.no_grad():
     for idx, (inputs,labels) in T(enumerate(valid_loader),total=len(valid_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        total += len(inputs)
        outputs = model(inputs.float())
        pred.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
        lab.extend(labels.cpu().numpy())        
        loss = criterion(outputs,labels)
        running_loss += loss.item()

   msg = 'Loss: {:.4f} \n Auc: {:.4f} '.format(running_loss/(len(valid_loader)), roc_auc_score(lab, pred))
   print(msg)
   lr_reduce_scheduler.step(running_loss)
   history.loc[epoch, 'valid_loss'] = running_loss/(len(valid_loader))
   history.loc[epoch, 'valid_auc'] = roc_auc_score(lab, pred)
   history.to_csv(os.path.join(history_dir, 'history_{}.csv'.format(model_name)), index=False)
   return  running_loss/(len(valid_loader))

# plist = [
        # {'params': model.features.parameters(),  'lr': learning_rate/50},
        # {'params': model.output.parameters(),  'lr': learning_rate},
        # {'params': model.backbone.layer2.parameters(),  'lr': learning_rate/50},
        # {'params': model.backbone.layer3.parameters(),  'lr': learning_rate/50},
        # {'params': model.backbone.layer4.parameters(),  'lr': learning_rate/50}
    # ]
# plist = [
#   {"params": model.head1.parameters(), "lr": learning_rate},
#   {"params": model.head2.parameters(), "lr": learning_rate},
#   {"params": model.head3.parameters(), "lr": learning_rate},
#   # {"params": model.backbone.extract_features.parameters(), "lr": learning_rate/100}
# ]
# optimizer = Over9000(plist, lr=learning_rate, weight_decay=1e-3)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, total_steps=None, epochs=n_epochs, steps_per_epoch=3348, pct_start=0.0,
                                  #  anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
criterion = nn.CrossEntropyLoss()

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  optimizer.load_state_dict(tmp['optim'])
  best_valid_recall = tmp['recall']
  prev_epoch_num = tmp['epoch']
  best_valid_loss = tmp['best_loss']
  del tmp
  print('Model Loaded!')

if apex:
    amp.initialize(model, optimizer, opt_level='O1')

for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())
    # stdscr = curses.initscr()
    train(epoch,history)
    valid_loss = evaluate(epoch,history)
    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}. Saving checkpoint')
        best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_reduce_scheduler.state_dict(), 'best_loss':valid_loss, 'epoch':epoch}
        torch.save(best_state, os.path.join(model_dir, model_name+'_loss.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, '{}_model_weights_best_loss.pth'.format(model_name))) ## Saving model weights based on best validation accuracy.
        best_valid_loss = valid_loss 