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
from utils import *
from optimizers import Over9000
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from model.seresnext import seresnext
from model.effnet import EffNet
# from model.densenet import *
## This library is for augmentations .
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    RandomCrop,
    Resize,
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
    VerticalFlip,
    HorizontalFlip,
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
    HueSaturationValue,
    Normalize, 
)
n_fold = 5
fold = 0
SEED = 24
batch_size = 50
sz = 384
learning_rate = 5e-4
patience = 4
opts = ['normal', 'mixup', 'cutmix']
device = 'cuda:0'
apex = True
pretrained_model = 'efficientnet-b3'
model_name = '{}_trial_stage1_fold_{}'.format(pretrained_model, fold)
model_dir = 'model_dir'
history_dir = 'history_dir'
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
load_model = False
history = pd.DataFrame()
prev_epoch_num = 0
n_epochs = 40
best_valid_loss = np.inf
best_valid_auc = 0.0
balanced_sampler = False
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

train_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    # GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.5),
    # RandomAugMix(severity=1, width=1, alpha=1., p=0.3),
    # OneOf([
    #     ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=30,border_mode=cv2.BORDER_CONSTANT,value =0),
    #     GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =0, p=0.1),
    #     OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =0)                  
    #     ], p=0.3),
    # OneOf([
    #     GaussNoise(var_limit=0.02),
    #     # Blur(),
    #     GaussianBlur(blur_limit=3),
    #     RandomGamma(p=0.8),
    #     ], p=0.5),
    HueSaturationValue(p=0.4),
    HorizontalFlip(0.4),
    VerticalFlip(0.4),
    Normalize(always_apply=True)
    ]
      )
val_aug = Compose([Normalize(always_apply=True)])
df = pd.read_csv('data/folds.csv')
X, y = df['image_id'], df['target']
# train_df['fold'] = np.nan
df= df.sample(frac=1, random_state=SEED).reset_index(drop=True)

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
df['age_approx'] /= df['age_approx'].max()
df['age_approx'] = df['age_approx'].fillna(0)
df['patient_id'] = df['patient_id'].fillna(0)
meta_features = ['sex', 'age_approx'] + [col for col in df.columns if 'site_' in col]
meta_features.remove('anatom_site_general_challenge')
#split data
# mskf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
# for i, (_, test_index) in enumerate(mskf.split(X, y)):
#     train_df.iloc[test_index, -1] = i
    
df['fold'] = df['fold'].astype('int')
idxs = [i for i in range(len(df))]
train_idx = []
val_idx = []
train_df = df[df['fold'] != fold]
valid_df = df[(df['fold'] == fold) & (df['source'] == 'ISIC20')]
train_meta = np.array(train_df[meta_features].values, dtype=np.float32)
valid_meta = np.array(valid_df[meta_features].values, dtype=np.float32)
# model = seresnext(pretrained_model).to(device)
model = EffNet(pretrained_model=pretrained_model, n_meta_features=train_meta.shape[1]).to(device)

train_ds = MelanomaDataset(train_df.image_id.values, train_meta, train_df.target.values, dim=sz, transforms=train_aug)
if balanced_sampler:
  print('Using Balanced Sampler....')
  train_loader = DataLoader(train_ds,batch_size=batch_size, sampler=BalanceClassSampler(labels=train_ds.get_labels(), mode="downsampling"), shuffle=False, num_workers=4)
else:
  train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, num_workers=4)

valid_ds = MelanomaDataset(valid_df.image_id.values, valid_meta, valid_df.target.values, dim=sz, transforms=val_aug)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=4)

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
  
  # if epoch<10:
  #   rate = 1
  # elif epoch>=10 and rate>0.60:
  #   rate = np.exp(-(epoch-10)/30)
  # else:
  #   rate = 0.60
  for idx, (inputs,meta,labels) in enumerate(train_loader):
    inputs = inputs.to(device)
    meta = meta.to(device)
    labels = labels.to(device)
    total += len(inputs)
    choice = choices(opts, weights=[1.00, 0.0, 0.0])
    optimizer.zero_grad()
    if choice[0] == 'normal':
      outputs = model(inputs.float(), meta)
      loss = ohem_loss(rate, criterion, outputs, labels)
      running_loss += loss.item()
    
    elif choice[0] == 'mixup':
      inputs, targets = mixup(inputs, labels, np.random.uniform(0.8, 1.0))
      outputs = model(inputs.float())
      loss = mixup_criterion(outputs, targets, criterion=criterion, rate=rate)
      running_loss += loss.item()
    
    elif choice[0] == 'cutmix':
      inputs, targets = cutmix(inputs, labels, np.random.uniform(0.8, 1.0))
      outputs = model(inputs.float())
      loss = cutmix_criterion(outputs, targets, criterion=criterion, rate=rate)
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
    if idx%5==0:
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
        lab.extend(torch.argmax(labels, 1).cpu().numpy())    
        loss = criterion(outputs,labels).mean()
        running_loss += loss.item()
   auc = roc_auc_score(lab, pred)
   msg = 'Loss: {:.4f} \n Auc: {:.4f} '.format(running_loss/(len(valid_loader)), auc)
   print(msg)
   lr_reduce_scheduler.step(running_loss)
   history.loc[epoch, 'valid_loss'] = running_loss/(len(valid_loader))
   history.loc[epoch, 'valid_auc'] = roc_auc_score(lab, pred)
   history.to_csv(os.path.join(history_dir, 'history_{}.csv'.format(model_name)), index=False)
   return running_loss/(len(valid_loader)), auc

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, total_steps=None, epochs=n_epochs, steps_per_epoch=3348, pct_start=0.0,
                                  #  anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)
lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
criterion = nn.BCEWithLogitsLoss()
# criterion = FocalLoss(logits=True).to(device)
# criterion = LabelSmoothing().to(device) 
# criterion = criterion_margin_focal_binary_cross_entropy

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  optimizer.load_state_dict(tmp['optim'])
  prev_epoch_num = tmp['epoch']
  best_valid_loss = tmp['best_loss']
  evaluate(-1,history)
  del tmp
  print('Model Loaded!')

if apex:
    amp.initialize(model, optimizer, opt_level='O1')

for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())
    # stdscr = curses.initscr()
    train(epoch,history)
    valid_loss, valid_auc = evaluate(epoch,history)
    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}. Saving checkpoint')
        best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_reduce_scheduler.state_dict(), 'best_loss':valid_loss, 'epoch':epoch}
        torch.save(best_state, os.path.join(model_dir, model_name+'_loss.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, '{}_model_weights_best_loss.pth'.format(model_name))) ## Saving model weights based on best validation accuracy.
        best_valid_loss = valid_loss
    if valid_auc>best_valid_auc:
        print(f'Validation auc has increased from:  {best_valid_auc:.4f} to: {valid_auc:.4f}. Saving checkpoint')
        best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_reduce_scheduler.state_dict(), 'best_auc':valid_auc, 'epoch':epoch}
        torch.save(best_state, os.path.join(model_dir, model_name+'_auc.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, '{}_model_weights_best_auc.pth'.format(model_name))) ## Saving model weights based on best validation accuracy.
        best_valid_auc = valid_auc 