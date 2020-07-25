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

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target.astype('int')] = 1.
    return vec

class MelanomaDataset(Dataset):
    def __init__(self, image_ids, meta_features=None, labels=None, dim=256, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms
        self.dim = dim
        self.meta_features = meta_features
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.dim, self.dim), interpolation = cv2.INTER_AREA)
        self.gen_challenge = np.zeros(7)
        self.gen_challenge[self.meta_features[idx][-1].astype('int')+1] = 1
        meta = self.meta_features[idx][:-1]
        meta = np.hstack((meta, self.gen_challenge))
        # image = image.astype(np.float32) / 255.0

        if self.transforms is not None:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        if self.labels is not None:
            target = self.labels[idx]
            return image, meta.astype('float32'), onehot(2, target)
        else:
            return image_id, image, meta.astype('float')

    def __len__(self):
        return len(self.image_ids)

    def get_labels(self):
        return list(self.labels)