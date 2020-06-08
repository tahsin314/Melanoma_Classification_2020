## Code copied from Lukemelas github repository have a look
## https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from .utils import *
from efficientnet_pytorch import EfficientNet

class EfficientNetWrapper(nn.Module):
    def __init__(self, pretrained_model='efficientnet-b4'):
        super(EfficientNetWrapper, self).__init__()
        # Load imagenet pre-trained model 
        self.backbone = EfficientNet.from_pretrained(pretrained_model, in_channels=3).to('cuda:0')
        self.backbone._fc = nn.Linear(in_features=1408, out_features=2, bias=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x