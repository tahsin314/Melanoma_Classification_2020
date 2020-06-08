import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
from random import choices
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, data_idxs, loc='data/512x512-dataset-melanoma/512x512-dataset-melanoma/', dim=256, transforms=None, phase='train'):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.data_idxs = data_idxs
        self.transforms = transforms
        self.phase = phase
        self.dim = dim
        self.ROOT_PATH = loc

    def __getitem__(self, idx: int):
        image_id = self.df['image_id'][self.data_idxs[idx]]
        image = cv2.imread(f'{self.ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.dim, self.dim))
        image = image.astype(np.float32) / 255.0
        
        if self.transforms is not None:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        
        if self.phase != 'test':
            target = self.df['target'][self.data_idxs[idx]]
            return image, target
        else:
            return image_id, image

    def __len__(self):
        return len(self.data_idxs)

