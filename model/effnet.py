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
from losses.triplet_loss import *
from efficientnet_pytorch import EfficientNet

class EffNet(nn.Module):
    def __init__(self, n_meta_features=9, pretrained_model='efficientnet-b4', use_meta=True, out_neurons=600, meta_neurons=150,freeze_upto=1):
        super(EffNet, self).__init__()
        # Load imagenet pre-trained model 
        self.backbone = EfficientNet.from_pretrained(pretrained_model, in_channels=3)
        self.num_named_param = 0
        # Dirty way of finding out number of named params
        for l, (name, param) in enumerate(self.backbone.named_parameters()):
            self.num_named_param = l
        self.freeze_upto_blocks(freeze_upto)
        in_features = self.backbone._fc.in_features
        self.out_neurons = out_neurons
        self.meta_neurons = meta_neurons
        self.backbone._fc = nn.Linear(in_features=in_features, out_features=self.out_neurons, bias=True)
        self.backbone._avg_pooling = GeM()
        self.use_meta = use_meta
        if self.use_meta:
            self.meta_fc = nn.Sequential(nn.Linear(n_meta_features, self.out_neurons ),
                                  nn.BatchNorm1d(self.out_neurons ),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(self.out_neurons , self.meta_neurons),  # FC layer output will have 750 features
                                  nn.BatchNorm1d(self.meta_neurons),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.4))
            self.output = nn.Linear(self.out_neurons  + self.meta_neurons, 2)
        else:
            self.backbone._avg_pooling = nn.Sequential(AdaptiveConcatPool2d(), Swish(), Flatten())
            # self.backbone._fc = nn.Sequential(*[bn_drop_lin(self.in_features*2, 512, True, 0.5, Swish()), bn_drop_lin(512, 2, True, 0.5)])
            # self.head = Head(in_features, 2, activation='mish', use_meta=self.use_meta)
            # self.output = nn.Linear(self.out_neurons, 2)
            self.backbone._fc = nn.Linear(in_features=2*in_features, out_features=2, bias=True)
        
    def forward(self, x, meta_data=None):
        if self.use_meta:
            cnn_features = self.backbone(x)
            meta_features = self.meta_fc(meta_data)
            features = torch.cat((cnn_features, meta_features), dim=1)
            output = self.output(features)
            return output
        else:
            x = self.backbone(x)
            # print(x.size())
            # output = self.head(x, meta_data)
            return x

    def freeze_upto_blocks(self, n_blocks):
        '''
        Freezes upto bottom n_blocks
        '''
        if n_blocks == -1:
            return

        num_freeze_params = 6 + 12*n_blocks
        for l, (name, param) in enumerate(self.backbone.named_parameters()):
            if not 'bn' in name and l<=self.num_named_param-num_freeze_params:
                param.requires_grad = False

class EffNet_ArcFace(nn.Module):
    def __init__(self, n_meta_features=9, pretrained_model='efficientnet-b4', use_meta=True, out_neurons=500, meta_neurons=250, freeze_upto=1):
        super(EffNet_ArcFace, self).__init__()
        # Load imagenet pre-trained model 
        self.backbone = EfficientNet.from_pretrained(pretrained_model, in_channels=3)
        self.num_named_param = 0
        # Dirty way of finding out number of named params
        for l, (name, param) in enumerate(self.backbone.named_parameters()):
            self.num_named_param = l
        self.freeze_upto_blocks(freeze_upto)
        in_features = self.backbone._fc.in_features
        self.out_neurons = out_neurons
        self.meta_neurons = meta_neurons
        self.backbone._fc = nn.Linear(in_features=in_features, out_features=self.out_neurons, bias=True)
        self.backbone._avg_pooling = GeM()
        self.use_meta = use_meta
        if self.use_meta:
            self.meta_fc = nn.Sequential(nn.Linear(n_meta_features, self.out_neurons),
                                  nn.BatchNorm1d(self.out_neurons),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(self.out_neurons, self.meta_neurons),  # FC layer output will have 750 features
                                  nn.BatchNorm1d(self.meta_neurons),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3))
            self.metric_classify = ArcMarginProduct(self.out_neurons+self.meta_neurons, 2)
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
    
    def freeze_upto_blocks(self, n_blocks):
        '''
        Freezes upto bottom n_blocks
        '''
        if n_blocks == -1:
            return

        num_freeze_params = 6 + 12*n_blocks
        for l, (name, param) in enumerate(self.backbone.named_parameters()):
            if not 'bn' in name and l<=self.num_named_param-num_freeze_params:
                param.requires_grad = False