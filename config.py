import os
import cv2
import pandas as pd
import torch
from torch import optim
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from augmentations.hair import Hair, AdvancedHairAugmentationAlbumentations
from augmentations.microscope import MicroscopeAlbumentations
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from model.seresnext import seresnext
from model.effnet import EffNet, EffNet_ArcFace
from utils import *
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
fold = 1
SEED = 24
batch_size = 8
sz = 456
learning_rate = 1e-4
patience = 3
accum_step = 48 // batch_size
opts = ['normal', 'mixup', 'cutmix']
choice_weights = [0.8, 0.1, 0.1]
device = 'cuda:0'
apex = False
pretrained_model = 'efficientnet-b5'
model_name = '{}_trial_stage1_fold_{}'.format(pretrained_model, fold)
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
TTA = 1
balanced_sampler = False
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
    MicroscopeAlbumentations(0.25),
    # RandomAugMix(severity=1, width=1, alpha=1., p=0.3),
    # OneOf([
    #     ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=30,border_mode=cv2.BORDER_CONSTANT,value =0),
    #     GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT,value =0, p=0.1),
    #     OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT,value =0)                  
    #     ], p=0.3),
    OneOf([
        # GaussNoise(var_limit=0.02),
        # Blur(),
        GaussianBlur(blur_limit=3),
        RandomGamma(p=0.8),
        ], p=0.5),
    HueSaturationValue(p=0.4),
    HorizontalFlip(0.4),
    VerticalFlip(0.4),
    Normalize(always_apply=True)
    ]
      )
val_aug = Compose([Normalize(always_apply=True)])
data_dir = 'data'
image_path = f'{data_dir}/512x512-dataset-melanoma/512x512-dataset-melanoma/'
test_image_path = f'{data_dir}/512x512-test/512x512-test'
pseduo_df = pd.read_csv('submissions/test_946.csv')
df = pd.read_csv(f'{data_dir}/folds.csv')
meta_features = ['sex', 'age_approx', 'site_head/neck', 'site_lower extremity', 'site_oral/genital', 'site_palms/soles', 'site_torso', 'site_upper extremity', 'site_nan']