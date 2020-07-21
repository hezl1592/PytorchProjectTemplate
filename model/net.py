# -*- coding: utf-8 -*-
# Time    : 2020/7/14 18:54
# Author  : zlich
# Filename: net.py
from model.backbone.mobilenetv2 import MobileNetV2_3feature
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)


class LaneAttrNet(nn.Module):
    def __init__(self, num_classes,
                 num_attributes,
                 in_channels=128,
                 fc_channels=3965):
        super(LaneAttrNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1,
                      padding=(4, 4), bias=False, dilation=(4, 4)),
            nn.BatchNorm2d(num_features=32, eps=1e-03),
            nn.ReLU()
        )

        self.layers_final = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,
                      out_channels=5,
                      kernel_size=(1, 1),
                      stride=1,
                      padding=(0, 0),
                      bias=True),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear1 = nn.Linear(fc_channels, 128)
        self.linear2 = nn.Linear(128, num_classes * num_attributes)

    def forward(self, x):
        r"""1x128x26x122 -> 1x5x13x61"""
        # inputs
        x = self.layers(x)
        x = self.layers_final(x)
        x = F.softmax(x, dim=1)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x.sigmoid()


class parsingNet(nn.Module):
    def __init__(self, num_lanes=4, backbone='mobilenetv2', cls_dim=(37, 10, 4),
                 use_aux=True):
        super(parsingNet, self).__init__()
        self.num_lanes = num_lanes
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        self.model = MobileNetV2_3feature()

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(64, 64, 3, padding=1),
                conv_bn_relu(64, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(96, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(1280, 320, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(320, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, num_lanes + 1, 1)
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        self.pool = torch.nn.Conv2d(1280, 8, 1)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1920, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4

        self.segmaxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.laneAtrr = LaneAttrNet(num_classes=4, num_attributes=8, in_channels=480, fc_channels=1200)

        initialize_weights(self.cls)
        initialize_weights(self.laneAtrr)

    def forward(self, x):
        x2, x3_, fea = self.model(x)

        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3_)
            x3 = torch.nn.functional.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4, size=x2.shape[2:], mode='bilinear', align_corners=True)
            aux_seg_ = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg_)
        else:
            aux_seg = None

        fea_ = self.pool(fea).view(-1, 1920)
        group_cls = self.cls(fea_).view(-1, *self.cls_dim)

        seg_map = self.segmaxpool2(aux_seg_)

        seg_map = torch.cat([x3_, seg_map], dim=1)
        type_cls = self.laneAtrr(seg_map)
        type_cls = type_cls.view(-1, 2, 16)

        if self.use_aux:
            return group_cls, aux_seg, type_cls

        return group_cls, type_cls


if __name__ == '__main__':
    from model.model_test_common import *

    model = parsingNet(cls_dim=(201, 32, 4), use_aux=True).cuda()

    from model.model_utils import modelParams_FLOPs, modelTime

    modelParams_FLOPs(model, inputTensor)
    modelTime(model, inputTensor)
    # modelParams_FLOPs()

    # net = MobileNetV2_3feature()
    # modelParams_FLOPs(net, inputTensor)
