# -*- coding: utf-8 -*-
# Time    : 2020/7/20 17:20
# Author  : zlich
# Filename: dataloder.py

import torch, os
import numpy as np
import torchvision.transforms as transforms
from data import BDD100K_Area_Seg
from torch.utils.data import DataLoader


def get_train_loader(batch_size, basePath, targetSize=(640, 360)):
    train_dataset = BDD100K_Area_Seg(base_dir=basePath, split='train', target_size=targetSize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    return train_loader


def get_val_loader(batch_size, basePath, targetSize=(640, 360)):
    valid_dataset = BDD100K_Area_Seg(base_dir=basePath, split='val', target_size=targetSize)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    return valid_loader


if __name__ == '__main__':
    test_loader = get_train_loader(16, '/home/hezl/laneDatasets/Bdd100k')
    for i, batch in enumerate(test_loader):
        print("[%2d/%d] | " % (i, len(test_loader)), end='')
        print(batch[0].shape, batch[1].shape)
        if i > 5:
            break
