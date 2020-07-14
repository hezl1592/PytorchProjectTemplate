# -*- coding: utf-8 -*-
# Time    : 2020/7/13 11:24
# Author  : zlich
# Filename: model_utils.py
import thop
import torch
import torchvision.models as models
import os
from utils.print_utils import print_info_message
import time
import math


def modelParams_FLOPs(model, inputTensor):
    model, inputTensor = (model.cuda(), inputTensor.cuda()) \
        if torch.cuda.is_available() else (model, inputTensor)

    input_size = tuple(inputTensor.shape[1:])
    flops, params = thop.profile(model=model, inputs=(inputTensor,))

    print_info_message("FLOPS: {:.4f} Million".format(flops / 1e6))
    print_info_message("Parms: {:.4f} Million".format(params / 1e6))
    print()


def modelTime(model, inputTensor):
    model, inputTensor = (model.cuda(), inputTensor.cuda()) \
        if torch.cuda.is_available() else (model, inputTensor)

    model.eval()
    timeList = []
    with torch.no_grad():
        for i in range(102):
            init_time = time.time()
            out = model(inputTensor)
            timeList.append(time.time() - init_time)

    timeList = timeList[1:-1]
    averageTime = sum(timeList) / len(timeList)
    print_info_message("Run {} times, average inference time: {:.5f}sec".format(len(timeList), averageTime))
    print()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    input_tensor = torch.rand(1, 3, 360, 640)
    model = models.vgg16()

    modelParams_FLOPs(model, input_tensor)
    modelTime(model, input_tensor)
