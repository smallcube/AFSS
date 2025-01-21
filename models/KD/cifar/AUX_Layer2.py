import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.autograd import Variable
import sys, os
import numpy as np
import random
from models.CosNormClassifier import CosNorm_Classifier


class Aux_Layer1(nn.Module):
    def __init__(self, inplanes, out_planes, num_classes=1000, groups=32, reduction=8, normalized=False, scale=30,
                 dropout=None):
        super(Aux_Layer1, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        #width = int(num_classes * (base_width / 64.)) * groups
        width = 2048

        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, width, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            #nn.Conv2d(width, width, kernel_size=3, padding=0, groups=1),
            #nn.SyncBatchNorm(width),
            #nn.ReLU(inplace=True),
            nn.Conv2d(width, out_planes, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

        channel = out_planes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, num_classes, scale=scale)
        else:
            self.FC1 = nn.Linear(channel, 2)
            self.FC2 = nn.Linear(2, num_classes)

    def forward(self, x):
        # x1 = self.conv(x)
        x2 = self.preBlock(x)
        x2 = F.relu(x2)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        feature = out
        if self.training:
            out = F.dropout(out)
        out1 = self.FC1(out)
        out = self.FC2(out1)

        if self.use_dropout:
            out = self.dropout(out)

        return out, feature


class Aux_Layer2(nn.Module):
    def __init__(self, inplanes, out_planes, num_classes=1000, groups=32, reduction=8, normalized=False, scale=30,
                 dropout=None):
        super(Aux_Layer2, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        #width = int(num_classes * (base_width / 64.)) * groups
        width = 2048

        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, width, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            #nn.Conv2d(width, width, kernel_size=3, padding=0, groups=1),
            #nn.SyncBatchNorm(width),
            #nn.ReLU(inplace=True),
            nn.Conv2d(width, out_planes, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

        '''
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, mid_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel), )
        '''

        channel = out_planes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, num_classes, scale=scale)
        else:
            self.FC1 = nn.Linear(channel, 2)
            self.FC2 = nn.Linear(2, num_classes)

    def forward(self, x):
        # x1 = self.conv(x)
        x2 = self.preBlock(x)
        
        #x3 = self.downsample(x)
        # print(x2.shape, "       ", x3.shape, "    ", x.shape)
        #x2 = x2 + x3
        x2 = F.relu(x2)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)
        

        out = self.avg(out)
        out = out.view(out.size(0), -1)

        feature = out

        
        if self.training:
            out = F.dropout(out)
        out1 = self.FC1(out)
        out = self.FC2(out1)

        if self.use_dropout:
            out = self.dropout(out)

        return out, feature
