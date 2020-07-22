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
try:
    import torch_xla.core.xla_model as xm
    _xla_available = True
except ImportError:
    _xla_available = False

from tqdm import tqdm_notebook as tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class MelanomaDataset:
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
        image = cv2.resize(image, (self.dim, self.dim))
        # image = image.astype(np.float32) / 255.0
        
        if self.transforms is not None:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        if self.labels is not None:
            target = self.labels[idx]
            return image, self.meta_features[idx], onehot(2, target)
        else:
            return image_id, image, self.meta_features[idx]

    def __len__(self):
        return len(self.image_ids)

    def get_labels(self):
        return list(self.labels)

class MelanomaDataloader:
    def __init__(self, image_ids, meta_features=None, labels=None, dim=256, transforms=None):
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms
        self.dim = dim
        self.meta_features = meta_features
        self.dataset = MelanomaDataset(
            image_ids=self.image_ids,
            meta_features = self.meta_features,
            labels=self.labels,
            dim=self.dim,
            transforms=self.transforms
        )

    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True, tpu=False):
        sampler = None
        if tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return data_loader