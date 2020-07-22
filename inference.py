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
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
# from model.densenet import *
from config import *
history = pd.DataFrame()
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
test_df = pd.read_csv('data/test.csv')
test_df= test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
test_image_path = 'data/512x512-test/512x512-test'
test_df['path'] = test_df['image_name'].map(lambda x: os.path.join(test_image_path,'{}.jpg'.format(x)))

test_df = meta_df(test_df, test_image_path)
test_meta = np.array(test_df[meta_features].values, dtype=np.float32)

model = EffNet(pretrained_model=pretrained_model, n_meta_features=len(meta_features)).to(device)

test_ds = MelanomaDataset(image_ids=test_df.path.values, meta_features=test_meta, dim=sz, transforms=test_aug)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

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
        # preds.extend(outputs.detach().cpu().numpy()[:,1])
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
submission['image_name'] = submission['image_name'].map(lambda x: x.replace(test_image_path, '').replace('.jpg', '').replace('/', ''))
submission.to_csv('submission.csv', index=False)