import os
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from torchvision.utils import save_image
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Cutout,
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
    RandomBrightness,
    RandomGamma,
    ShiftScaleRotate ,
    GaussNoise,
    Blur,
    MotionBlur,   
    GaussianBlur,
    Normalize, 
)
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
# from augmentations.cutouts import Cutout
from MelanomaDataset import *
from utils import *

sz = 256
train_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
    # GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.5),
    RandomAugMix(severity=1, width=1, alpha=1., p=0.3),
    HorizontalFlip(0.4),
    VerticalFlip(0.4),
    # Normalize(always_apply=True)
    ]
      )

def visualize(original_image):
    fontsize = 18
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(8,8,1)
    plt.axis('off')
    ax.imshow(original_image, cmap='gray')
    for i in range(63):
        augment = train_aug(image = image)
        aug_img = augment['image']
        ax = fig.add_subplot(8,8,i+2)
        plt.axis('off')
        ax.imshow(aug_img, cmap='gray')
    fig.savefig('aug.png')

train_df = pd.read_csv('data/folds.csv')
train_ds = MelanomaDataset(train_df.image_id.values, train_df.target.values, dim=sz, transforms=train_aug)
train_loader = DataLoader(train_ds,batch_size=64, shuffle=True)
im, _ = iter(train_loader).next()
print(im.size(), torch.max(im))
save_image(im, 'Aug.png', nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)