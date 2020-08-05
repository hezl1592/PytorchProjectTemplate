# -*- coding: utf-8 -*-
# Time    : 2020/8/4 18:53
# Author  : zlich
# Filename: export_onnx.py
import torch
import torch.nn.functional as F
import onnxruntime
import onnx
import time
from model import Deeplabv3plus_Mobilenet
from utils.train_utils import deviceSetting, savePath
from utils.test_utils import modelDeploy, ImageGet
from getargs import getArgs_
import sys
import argparse


class vismodel(Deeplabv3plus_Mobilenet):
    def __init__(self, output_channels=19, output_stride=8):
        super(vismodel, self).__init__(output_channels, output_stride)

    def forward(self, inputs):
        x, low_level_feat = self.encoder(inputs)
        x = self.spp(x)
        x = self.decoder(x, low_level_feat)
        x = self.logits(x)

        return F.interpolate(x, size=(360, 640), mode='bilinear', align_corners=True)


def transform_to_onnx(model, batch_size=1, IN_IMAGE_H=360, IN_IMAGE_W=640):
    x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).cuda()
    onnx_file_name = "onnx_{}_3_{}_{}.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)

    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      input_names=['input'], output_names=['output'], opset_version=11)

    print('Onnx model exporting done')
    return onnx_file_name


if __name__ == '__main__':
    args = argparse.Namespace()
    args.modelpath = '../../checkpoints/deeplabv3+_20200802-140613/deeplabv3+_640_360_best_37_0.791740.pth'
    _, args.device = deviceSetting(device="0")
    model = vismodel(3, 16)
    model = modelDeploy(args=args, model=model).to(args.device)
    print(torch.version.__version__)
    transform_to_onnx(model)
