# -*- coding: utf-8 -*-
# Time    : 2020/7/15 18:17
# Author  : zlich
# Filename: deeplabv3+.py

import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict


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
    def __init__(self, in_channels=2048, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError
        # dilations = [6, 12, 18]

        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = separableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = separableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = separableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)

        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)
        del x0, x1, x2, x3

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class Deeplabv3plus_Mobilenet(nn.Module):
    def __init__(self, output_channels=19, enc_type='xception65', dec_type='aspp', output_stride=8):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type

        self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=False)
        if enc_type == 'mobilenetv2':
            self.spp = create_mspp(dec_type)
        else:
            self.spp, self.decoder = create_spp(dec_type, output_stride=output_stride)
        self.logits = nn.Conv2d(256, output_channels, 1)

    def forward(self, inputs):
        #print("...", inputs.size())
        if self.enc_type == 'mobilenetv2':
            x = self.encoder(inputs)
            # print(r"F:\PythonLearn\pytorch-segmentation-master\src\models\net.py")
            #print("fea:\t", x.size())
            x = self.spp(x)
            # print("x1:\t", x.size())
            x = self.logits(x)
            # print(x.size())
            return x
        else:
            x, low_level_feat = self.encoder(inputs)
            x = self.spp(x)
            x = self.decoder(x, low_level_feat)
            x = self.logits(x)
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def get_1x_lr_params(self):
        for p in self.encoder.parameters():
            yield p

    def get_10x_lr_params(self):
        modules = [self.spp, self.logits]
        if hasattr(self, 'decoder'):
            modules.append(self.decoder)

        for module in modules:
            for p in module.parameters():
                yield p


if __name__ == '__main__':
    net = ASPP()
    print(net.state_dict().keys())