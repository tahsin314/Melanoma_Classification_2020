from copy import deepcopy
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torchvision import models
from pytorchcv.model_provider import get_model as ptcv_get_model
from .utils import get_cadene_model
from typing import Optional
from .utils import *
import timm
from pprint import pprint

class Resnest(nn.Module):

    def __init__(self, model_name='resnest50d_1s4x24d', use_meta=True, out_neurons=600, meta_neurons=150):
        super().__init__()
        self.backbone = timm.create_model(model_name)
        self.use_meta = use_meta
        self.in_features = 2048
        self.head = Head(self.in_features,2, activation='mish', use_meta=self.use_meta)

    def forward(self, x, meta_data=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        
        x = self.backbone.layer4(x)
        
        x = self.head(x, meta_data)
        return x
