# -*- coding: utf-8 -*-
# Time    : 2020/7/14 19:19
# Author  : zlich
# Filename: model_test_common.py
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
inputTensor = torch.rand(1, 3, 244, 244)


def printNet(model):
    print(model)
