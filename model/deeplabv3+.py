# -*- coding: utf-8 -*-
# Time    : 2020/7/15 18:17
# Author  : zlich
# Filename: deeplabv3+.py

import torch
import torch.nn as nn


class separableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True):
        super().__init__()
        self.depthWise = nn.Conv2d(inplanes, inplanes, kernel_size,
                                   stride=stride, padding=dilation,
                                   dilation=dilation, groups=inplanes, bias=False)
        self.bnDepth = nn.BatchNorm2d(inplanes)
        self.pointWise = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bnPoint = nn.BatchNorm2d(planes)

        if relu_first:
            self.block = nn.Sequential(nn.ReLU(),
                                       self.depthWise,
                                       self.bnDepth,
                                       self.pointWise,
                                       self.bnPoint)
        else:
            self.block = nn.Sequential(self.depthWise,
                                       self.bnDepth,
                                       nn.ReLU(),
                                       self.pointWise,
                                       self.bnPoint,
                                       nn.ReLU())

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, inChannel=2048, outChannel=256, outputStride=8):
        super(ASPP, self).__init__()
        if outputStride == 16:
            dilations = [6, 12, 18]
        elif outputStride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError
        self.dilations = dilations

        self.aspp0 = nn.Conv2d(inChannel, )
