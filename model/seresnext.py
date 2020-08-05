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

class seresnext(nn.Module):

    def __init__(self, model_name='seresnext50_32x4d', use_meta=True, out_neurons=600, meta_neurons=150):
        super().__init__()
        self.backbone = get_cadene_model('se_resnext50_32x4d')
#         in_features = self.backbone.fc.in_features
        # for Resnet
        # print(self.backbone.layer4)
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # arch = ptcv_get_model(model_name, pretrained=True)
        # print(arch)
        # arch_layer0 = list(arch.layer0.children())
        # arch_layer4 = list(arch.layer4.children())
        # w = arch_layer0[0].weight
        # self.backbone.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.backbone.layer0.conv1.weight = nn.Parameter(torch.sum(w, dim=1, keepdim=True))
        self.backbone.layer0.relu1 = Mish()
        self.use_meta = use_meta
        # nc = 2048 # 512 if res34
        # nc = self.backbone.layer4.conv3.weight.shape[0]
        # print(self.backbone.layer4)
        self.in_features = 2048
        
        self.head = Head(self.in_features,2, activation='mish', use_meta=self.use_meta)
        # self.backbone.output = nn.Linear(nc, 2)
        
        to_Mish(self.backbone.layer1), to_Mish(self.backbone.layer2), to_Mish(self.backbone.layer3)
        to_Mish(self.backbone.layer4)
        
    def forward(self, x, meta_data=None):
        x = self.backbone.layer0.conv1(x)
        x = self.backbone.layer0.bn1(x)
        x = self.backbone.layer0.relu1(x)
        x = self.backbone.layer0.pool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        
        x = self.backbone.layer4(x)
        
        x = self.head(x, meta_data)
        return x