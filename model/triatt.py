'''
https://arxiv.org/pdf/1903.06150.pdf
'''
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from .utils import *
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import timm
from efficientnet_pytorch import EfficientNet
from pprint import pprint
# import att_grid_generator_cuda¿

class Atthead(nn.Module):
    expansion = 4

    def __init__(self, att = False):
        super(Atthead, self).__init__()
        self.att = att
        self.in_channels = 1536
        self.out_channels = 1536
        self.kernel_size = _pair(3)
        self.weight1 = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.weight2 = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.relu1_1 = Swish()
        self.relu1_2 = Swish()
        self.relu2_1 = Swish()
        self.relu2_2 = Swish()
        self.relu2_3 = Swish()

        self.reset_parameters()



    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if not self.att:
            return x

        att1_1 = F.conv2d(x, self.weight1, bias=None, stride=1, padding=1, dilation = 1)
        att1_1 = self.relu1_1(att1_1)
        att1_2 = F.conv2d(x, self.weight1, bias=None, stride=1, padding=2, dilation = 2)
        att1_2 = self.relu1_2(att1_2)
        att1 = att1_1 + att1_2
        att2_1 = F.conv2d(att1, self.weight2, bias=None, stride=1, padding=1, dilation = 1)
        att2_1 = self.relu2_1(att2_1)
        att2_2 = F.conv2d(att1, self.weight2, bias=None, stride=1, padding=2, dilation = 2)
        att2_2 = self.relu2_2(att2_2)
        att2_3 = F.conv2d(att1, self.weight2, bias=None, stride=1, padding=3, dilation = 3)
        att2_3 = self.relu2_3(att2_3)
        att2 = att2_1 + att2_2 + att2_3

        return att2

class tri_att(nn.Module):
  def __init__(self):
    super(tri_att, self).__init__()
    self.feature_norm = nn.Softmax(dim=2)
    self.bilinear_norm = nn.Softmax(dim=2)


  def forward(self, x):
    n = x.size(0)
    c = x.size(1)
    h = x.size(2)
    w = x.size(3)
    f = x.reshape(n, c, -1)

    # *7 to obtain an appropriate scale for the input of softmax function.
    f_norm = self.feature_norm(f * 7)

    bilinear = f_norm.bmm(f.transpose(1, 2))
    bilinear = self.bilinear_norm(bilinear)
    trilinear_atts = bilinear.bmm(f).view(n, c, h, w).detach()
    structure_att = torch.sum(trilinear_atts,dim=1, keepdim=True)
    
    return structure_att

def att_sample(data, structure_att, out_size):
    n = data.size(0)
    h = data.size(2)
    structure_att = F.interpolate(structure_att, (h, h), mode='bilinear', align_corners=False).squeeze(1)
    map_sx, _ = torch.max(structure_att, 2)
    map_sx = map_sx.unsqueeze(2)
    map_sy, _ = torch.max(structure_att, 1)
    map_sy = map_sy.unsqueeze(2)
    sum_sx = torch.sum(map_sx, (1, 2), keepdim=True)
    sum_sy = torch.sum(map_sy, (1, 2), keepdim=True)
    map_sx = torch.div(map_sx, sum_sx)
    map_sy = torch.div(map_sy, sum_sy)
    map_xi = torch.zeros_like(map_sx)
    map_yi = torch.zeros_like(map_sy)
    index_x = torch.zeros((n, out_size, 1)).cuda()
    index_y = torch.zeros((n, out_size, 1)).cuda()
    # att_grid_generator_cuda.forward(map_sx, map_sy, map_xi, map_yi,
    #                                 index_x, index_y,
    #                                 h, out_size, 4,
    #                                 5, out_size/h)
    one_vector = torch.ones_like(index_x)
    grid_x = torch.matmul(one_vector, index_x.transpose(1, 2)).unsqueeze(-1)
    grid_y = torch.matmul(index_y, one_vector.transpose(1, 2)).unsqueeze(-1)
    grid = torch.cat((grid_x, grid_y), 3).float()
    structure_data = F.grid_sample(data, grid)
    return structure_data


class Resnest(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', attn=False):
        super().__init__()
        # self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        self.in_features = 2048
        # self.head = Head(self.in_features,2, activation='mish', use_meta=self.use_meta)
        self.relu = Mish()
        self.maxpool = GeM()
        self.attn = attn
        self.head = Atthead(self.attn)
        # self.attn1 = AttentionBlock(256, 1024, 512, 4, normalize_attn=normalize_attn)
        # self.attn2 = AttentionBlock(512, 1024, 512, 2, normalize_attn=normalize_attn)
        # self.output1 = nn.Linear(770, 128)
        # self.output = nn.Linear(128, 2)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        
        out = self.head(layer4)
        return out

class Attn_EfficientNet(nn.Module):

    def __init__(self, model_name='efficientnet_b0', attn=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        # self.backbone = EfficientNet.from_pretrained(model_name, in_channels=3)
        # print(self.backbone)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(self.in_features, 128)
        self.output = nn.Linear(128, 2)
        self.attn = attn
        self.head = Atthead(self.attn)

    def forward(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        # print(x.size())
        # x = self.backbone.global_pool(x)
        # print(x.size())
        x = self.head(x)
        # x = x.flatten(1)
        # x = self.backbone.classifier(x)
        # x = self.output(x)
        return x


class Tasn(nn.Module):

    def __init__(self):
        super(Tasn, self).__init__()
        self.model_att = Attn_EfficientNet(model_name='mixnet_l', attn=True)
        self.model_cls = Attn_EfficientNet(model_name='mixnet_xl', attn=False)
        self.trilinear_att = tri_att()
        self.pool_att = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_att = nn.Linear(1536, 128)
        self.output_att = nn.Linear(128, 2)
        self.pool_cls = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(1536, 256)
        self.output_cls = nn.Linear(256, 2)


    def forward(self, x):
        n = x.size(0)
        c = x.size(1)
        w = x.size(2)

        input_att = F.interpolate(x, (380, 380), mode='bilinear', align_corners=False)
        conv_att = self.model_att(input_att)
        att = self.trilinear_att(conv_att)
        input_cls = att_sample(x,att,380)
        conv_cls = self.model_cls(input_cls)
        out_att = self.pool_att(conv_att)
        out_att = torch.flatten(out_att, 1)
        out_att = self.fc_att(out_att)
        out_att = self.output_att(out_att)
        out_cls = self.pool_cls(conv_cls)
        out_cls = torch.flatten(out_cls, 1)
        # print(out_cls.size())
        out_cls = self.fc_cls(out_cls)
        out_cls = self.output_cls(out_cls)
        return out_att, out_cls
