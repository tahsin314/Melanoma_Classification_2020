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
batch_size = 32
sz = 320
learning_rate = 5e-4
patience = 5
opts = ['normal', 'mixup', 'cutmix']
device = 'cuda:0'
apex = True
pretrained_model = 'efficientnet-b3'
model_name = '{}_trial_stage1_fold_{}'.format(pretrained_model, fold)
model_dir = 'model_dir'
history_dir = 'history_dir'
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
load_model = True
history = pd.DataFrame()
prev_epoch_num = 0
n_epochs = 3
valid_recall = 0.0
best_valid_recall = 0.0
best_valid_loss = np.inf
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

test_aug = Compose([Normalize()])
test_df = pd.read_csv('data/sample_submission.csv')
# model = seresnext(pretrained_model).to(device)
model = EffNet(pretrained_model).to(device)

test_ds = MelanomaDataset(test_df.image_id.values, loc='data/512x512-test/512x512-test', transforms=test_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

## This function for train is copied from @hanjoonchoe
## We are going to train and track accuracy and then evaluate and track validation accuracy

def evaluate():
   model.eval()
   img_ids = []
   preds = []
   with torch.no_grad():
     for idx, (img_id, inputs) in T(enumerate(test_loader),total=len(test_loader)):
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        img_ids.extend(img_id)        
        preds.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())     
   return img_ids, preds

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
  model.load_state_dict(tmp['model'])
  print("Best Loss: {:4f}".format(tmp['best_loss']))
  del tmp
  print('Model Loaded!')

if apex:
    amp.initialize(model, opt_level='O1')

IMG_IDS, TARGET_PRED = evaluate()
zippedList =  list(zip(IMG_IDS, TARGET_PRED))
submission = pd.DataFrame(zippedList, columns = ['image_name','target'])
submission.to_csv('submission.csv', index=False)