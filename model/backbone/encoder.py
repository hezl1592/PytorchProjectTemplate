# -*- coding: utf-8 -*-
# Time    : 2020/7/22 18:44
# Author  : zlich
# Filename: encoder.py
from .xception import Xception65


def create_encoder(outstride=8):
    return Xception65(outstride)


if __name__ == '__main__':
    model = create_encoder()
    print(model.state_dict().keys())