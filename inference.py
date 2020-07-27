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
tta_aug1 = Compose([
  ShiftScaleRotate(p=1,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    Normalize(always_apply=True)])
tta_aug2 = Compose([
  Cutout(p=1.0, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    Normalize(always_apply=True)])
tta_aug3 = Compose([
  RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=1.0),
    Normalize(always_apply=True)])
tta_aug4 = Compose([
  AdvancedHairAugmentationAlbumentations(p=1.0),
    Normalize(always_apply=True)])
tta_aug5 = Compose([
  GaussianBlur(blur_limit=3, p=1),
    Normalize(always_apply=True)])
tta_aug6 = Compose([
  HueSaturationValue(p=1.0),
    Normalize(always_apply=True)])
tta_aug7 = Compose([
  HorizontalFlip(1.0),
    Normalize(always_apply=True)])

tta_aug8 = Compose([
  VerticalFlip(1.0),
    Normalize(always_apply=True)])

tta_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    ], p=0.20),
    RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.5),
    HueSaturationValue(p=0.4),
    HorizontalFlip(0.4),
    VerticalFlip(0.4),
    Normalize(always_apply=True)
    ]
      )
test_df = pd.read_csv('test_768v2.csv')
test_image_path = 'data/test_768'
# test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna(-1)
# # Sex features
# test_df['sex'] = test_df['sex'].fillna(-1)
# test_df['age_approx'] = test_df['age_approx'].fillna(0)
# test_df['patient_id'] = test_df['patient_id'].fillna(0)
test_meta = np.array(test_df[meta_features].values, dtype=np.float32)

model = EffNet(pretrained_model=pretrained_model).to(device)


# augs = [test_aug, tta_aug1, tta_aug2, tta_aug3, tta_aug4, tta_aug5, tta_aug6, tta_aug7, tta_aug8, tta_aug9]
augs = [test_aug, tta_aug1, tta_aug3, tta_aug6, tta_aug7, tta_aug8]
def evaluate():
   model.eval()
   PREDS = np.zeros((len(test_df), 1))
   with torch.no_grad():
     for t in range(TTA):
      print('TTA {}'.format(t+1))
      test_ds = MelanomaDataset(image_ids=test_df.image_name.values, meta_features=test_meta, dim=sz, transforms=augs[t])
      test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

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
  tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
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