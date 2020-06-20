## Code copied from Lukemelas github repository have a look
## https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from .utils import *
from losses.arcface import ArcMarginProduct
from efficientnet_pytorch import EfficientNet

class EffNet(nn.Module):
    def __init__(self, n_meta_features, pretrained_model='efficientnet-b4', use_meta=True):
        super(EffNet, self).__init__()
        # Load imagenet pre-trained model 
        self.backbone = EfficientNet.from_pretrained(pretrained_model, in_channels=3).to('cuda:0')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features=in_features, out_features=500, bias=True)
        self.backbone._avg_pooling = GeM()
        self.use_meta = use_meta
        if self.use_meta:
            self.meta_fc = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
            self.output = nn.Linear(500 + 250, 2)
        else:
            self.backbone._fc = nn.Linear(in_features=in_features, out_features=2, bias=True)
        
    def forward(self, x, meta_data=None):
        if self.use_meta:
            cnn_features = self.backbone(x)
            meta_features = self.meta_fc(meta_data)
            features = torch.cat((cnn_features, meta_features), dim=1)
            output = self.output(features)
            return output
        else:
            x = self.backbone(x)
            return x

class EffNet_ArcFace(nn.Module):
    def __init__(self, n_meta_features, pretrained_model='efficientnet-b4', use_meta=True):
        super(EffNet_ArcFace, self).__init__()
        # Load imagenet pre-trained model 
        self.backbone = EfficientNet.from_pretrained(pretrained_model, in_channels=3).to('cuda:0')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features=in_features, out_features=500, bias=True)
        self.backbone._avg_pooling = GeM()
        self.use_meta = use_meta
        if self.use_meta:
            self.meta_fc = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
            self.metric_classify = ArcMarginProduct(500+250, 2)
        else:
            self.backbone._fc = nn.Linear(in_features=in_features, out_features=2, bias=True)
        
    def forward(self, x, meta_data=None):
        if self.use_meta:
            cnn_features = self.backbone(x)
            meta_features = self.meta_fc(meta_data)
            features = torch.cat((cnn_features, meta_features), dim=1)
            output = self.metric_classify(features)
            return output
        else:
            x = self.backbone(x)
            return x