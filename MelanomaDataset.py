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
    def __init__(self, csv_file, data_idxs, transforms=None, phase='train'):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.data_idxs = data_idxs
        self.transforms = transforms
        self.phase = phase
        self.dim = 256

    def __getitem__(self, idx: int):
        TRAIN_ROOT_PATH = 'data/jpeg/train'
        image_id = self.df['image_name'][self.data_idxs[idx]]
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.dim, self.dim))
        image = image.astype(np.float32) / 255.0

        label = self.df['target'][self.data_idxs[idx]]
        if self.transforms:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        # target = onehot(2, label)
        target = label
        return image, target

    def __len__(self) -> int:
        return len(self.data_idxs)

    def get_labels(self):
        return list(self.labels)


