# -*- coding: utf-8 -*-
# Time    : 2020/7/14 18:55
# Author  : zlich
# Filename: mobilenetv2.py
'''
modified from https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py
'''
import torch.nn as nn
import math
import torch


def Conv_3x3_bn(inChannel, outChannel, stride):
    return nn.Sequential(
        nn.Conv2d(inChannel, outChannel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outChannel),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1_bn(inChannel, outChannel):
    return nn.Sequential(
        nn.Conv2d(inChannel, outChannel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outChannel),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class bottleneck(nn.Module):
    def __init__(self, inputChannel, config):
        super(bottleneck, self).__init__()
        t, c, n, s = config
        outputChannel = c
        layers = []
        for i in range(n):
            layers.append(InvertedResidual(inputChannel, outputChannel, s if i == 0 else 1, t))
            inputChannel = outputChannel
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, out_stride=32):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        if out_stride == 8:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 1],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
        if out_stride == 16:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
        input_channel = 32
        self.baselayer = Conv_3x3_bn(3, input_channel, 2)
        self.bottleneck1 = bottleneck(32, self.cfgs[0])
        self.bottleneck2 = bottleneck(16, self.cfgs[1])
        self.bottleneck3 = bottleneck(24, self.cfgs[2])
        self.bottleneck4 = bottleneck(32, self.cfgs[3])
        self.bottleneck5 = bottleneck(64, self.cfgs[4])
        self.bottleneck6 = bottleneck(96, self.cfgs[5])
        self.bottleneck7 = bottleneck(160, self.cfgs[6])

        # self.conv = Conv_1x1_bn(320, 1280)
        self._initialize_weights()

    def forward(self, x):
        x = self.baselayer(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        # x = self.conv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2_2Feature(MobileNetV2):
    def __init__(self, out_stride=16):
        super(MobileNetV2_2Feature, self).__init__(out_stride)

    def forward(self, x):
        x = self.baselayer(x)
        x = self.bottleneck1(x)
        x_4 = self.bottleneck2(x)
        x = self.bottleneck3(x_4)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        return x, x_4


class MobileNetV2_3feature(MobileNetV2):
    def __init__(self):
        super(MobileNetV2_3feature, self).__init__()

    def forward(self, x):
        x = self.baselayer(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x_8 = self.bottleneck3(x)
        x = self.bottleneck4(x_8)
        x_16 = self.bottleneck5(x)
        x = self.bottleneck6(x_16)
        x = self.bottleneck7(x)
        # x = self.conv(x)
        return x_8, x_16, x


if __name__ == '__main__':
    from model.model_test_common import *

    inputTensor = torch.rand(1, 3, 720, 1280)
    # inputTensor = torch.rand(1, 3, 224, 224)
    model = MobileNetV2_3feature()
    from model.model_utils import modelParams_FLOPs, modelTime

    # net = MobileNetV2_3feature()
    modelParams_FLOPs(model, inputTensor)

    net = MobileNetV2()
    modelParams_FLOPs(net, inputTensor)
    modelTime(net, inputTensor)

    net_8 = MobileNetV2(out_stride=8)
    modelParams_FLOPs(net_8, inputTensor)
    modelTime(net_8, inputTensor)

    net_16 = MobileNetV2(out_stride=16)
    modelParams_FLOPs(net_16, inputTensor)
    modelTime(net_16, inputTensor)
