import os
import cv2
import pandas as pd
import torch
from torch import optim
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from augmentations.hair import Hair, AdvancedHairAugmentationAlbumentations
from augmentations.microscope import MicroscopeAlbumentations
from augmentations.color_constancy import ColorConstancy
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
from utils import *
from albumentations.augmentations.transforms import Equalize, Posterize, Downscale
from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, CenterCrop,    
    RandomCrop, Resize, Crop, Compose, HueSaturationValue,
    Transpose, RandomRotate90, ElasticTransform, GridDistortion, 
    OpticalDistortion, RandomSizedCrop, Resize, CenterCrop,
    VerticalFlip, HorizontalFlip, OneOf, CLAHE, Normalize,
    RandomBrightnessContrast, Cutout, RandomGamma, ShiftScaleRotate ,
    GaussNoise, Blur, MotionBlur, GaussianBlur, 
)
n_fold = 5
fold = 0
SEED = 24
batch_size = 24
sz = 512
learning_rate = 3e-3
patience = 3
accum_step = 50 // batch_size
opts = ['normal', 'mixup', 'cutmix']
choice_weights = [0.80, 0.10, 0.10]
device = 'cuda:0'
mixed_precision = True
use_meta = False
pretrained_model = 'mixnet_xl'
model_name = f'{pretrained_model}_dim_{sz}'
# model_name = 'efficientnet-b6_trial_stage1_fold_0'
model_dir = 'model_dir'
history_dir = 'history_dir'
load_model = True
freeze_upto = -1 # Freezes upto bottom n_blocks
if load_model and os.path.exists(os.path.join(history_dir, f'history_{model_name}.csv')):
    history = pd.read_csv(os.path.join(history_dir, f'history_{model_name}.csv'))
else:
    history = pd.DataFrame()

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
n_epochs = 60
TTA = 6
balanced_sampler = True
pseudo_lo_thr = 0.10
pseudo_up_thr = 0.70

train_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    # GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.5),
    AdvancedHairAugmentationAlbumentations(p=0.3),
    MicroscopeAlbumentations(0.1),
    # RandomAugMix(severity=3, width=1, alpha=1., p=0.3),
    OneOf([
        # Equalize(p=0.2),
        Posterize(num_bits
        =4, p=0.4),
        Downscale(0.40, 0.80, cv2.INTER_LINEAR, p=0.3)                  
        ], p=0.2),
    OneOf([
        GaussNoise(var_limit=0.1),
        Blur(),
        GaussianBlur(blur_limit=3),
        # RandomGamma(p=0.7),
        ], p=0.3),
    HueSaturationValue(p=0.4),
    HorizontalFlip(0.4),
    VerticalFlip(0.4),
    # ColorConstancy(p=0.3, always_apply=False),
    Normalize(always_apply=True)
    ]
      )
val_aug = Compose([Normalize(always_apply=True)])
data_dir = 'data'
image_path = f'{data_dir}/train_768'
test_image_path = f'{data_dir}/test_768'
pseduo_df = pd.read_csv('submissions/sub_946.csv')
# df = pd.read_csv(f'{data_dir}/folds.csv')
gen_challenge = {'lower extremity': 2, 'torso':3, 'head/neck':0, 'oral/genital':5, 'palms/soles':4, 'upper extremity':1}
# meta_features = ['sex', 'age_approx', 'site_head/neck', 'site_lower extremity', 'site_oral/genital', 'site_palms/soles', 'site_torso', 'site_upper extremity', 'site_nan']
meta_features = ['sex','age_approx','anatom_site_general_challenge']
