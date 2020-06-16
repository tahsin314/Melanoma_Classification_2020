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
batch_size = 32
sz = 384
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
valid_recall = 0.0
best_valid_recall = 0.0
best_valid_loss = np.inf
TTA = 4
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

test_aug = Compose([Normalize()])
tta_aug =Compose([
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
df = pd.read_csv('data/train.csv')      
test_df = pd.read_csv('data/test.csv')
test_df= test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

'''
Meta features: https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
'''
# One-hot encoding of anatom_site_general_challenge feature
concat = test_df['anatom_site_general_challenge']
dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
test_df = pd.concat([test_df, dummies.iloc[:test_df.shape[0]]], axis=1)

# Sex features
test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
test_df['sex'] = test_df['sex'].fillna(-1)

# Age features
test_df['age_approx'] /= test_df['age_approx'].max()
test_df['age_approx'] = test_df['age_approx'].fillna(0)
test_df['patient_id'] = test_df['patient_id'].fillna(0)
test_df['site_lateral torso'] = np.nan
test_df['site_unknown'] = np.nan
test_df['site_lateral torso'] = test_df['site_lateral torso'].fillna(0)
test_df['site_unknown'] = test_df['site_unknown'].fillna(0)

meta_features = ['sex', 'age_approx'] + [col for col in test_df.columns if 'site_' in col]
meta_features.remove('anatom_site_general_challenge')
meta_features = ['sex', 'age_approx', 'site_head/neck', 'site_lateral torso', 'site_lower extremity', 'site_oral/genital', 'site_palms/soles', 'site_torso', 'site_unknown', 'site_upper extremity', 'site_nan']  
idxs = [i for i in range(len(df))]
train_idx = []
val_idx = []
test_meta = np.array(test_df[meta_features].values, dtype=np.float32)

model = EffNet(pretrained_model=pretrained_model, n_meta_features=len(meta_features)).to(device)

test_ds = MelanomaDataset(image_ids=test_df.image_name.values, meta_features=test_meta, loc='data/512x512-test/512x512-test', dim=sz, transforms=tta_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

def evaluate():
   model.eval()
   PREDS = np.zeros((len(test_loader.dataset), 1))
   with torch.no_grad():
     for t in range(TTA):
      print('TTA {}'.format(t+1))
      img_ids = []
      preds = []
      for idx, (img_id, inputs, meta) in T(enumerate(test_loader),total=len(test_loader)):
        inputs = inputs.to(device)
        meta = meta.to(device)
        outputs = model(inputs.float(), meta)
        img_ids.extend(img_id)        
        preds.extend(torch.softmax(outputs,1)[:,1].detach().cpu().numpy())
      PREDS += np.array(preds).reshape(len(test_loader.dataset), 1)
     PREDS /= TTA     
   return img_ids, list(PREDS[:, 0])

if load_model:
  tmp = torch.load(os.path.join(model_dir, model_name+'_auc.pth'))
  model.load_state_dict(tmp['model'])
  print("Best Loss: {:4f}".format(tmp['best_auc']))
  del tmp
  print('Model Loaded!')

if apex:
    amp.initialize(model, opt_level='O1')

IMG_IDS, TARGET_PRED = evaluate()
zippedList =  list(zip(IMG_IDS, TARGET_PRED))
submission = pd.DataFrame(zippedList, columns = ['image_name','target'])
submission.to_csv('submission.csv', index=False)