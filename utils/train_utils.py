# -*- coding: utf-8 -*-
# Time    : 2020/7/15 13:32
# Author  : zlich
# Filename: train_utils.py
import torch
import os
# from utils.print_utils import *
from utils.log_utils import infoLogger
import time


def deviceSetting(logger):
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    if num_gpus >= 1:
        logger.info("GPU found, device: {}, number: {}.".format(device, num_gpus))
    else:
        logger.warning("No GPU found, device: CPU.")
        # print_warning_message()
    return num_gpus, device


def savePath(args):
    saveDir = args.log_dir
    childPath = '{}_{}'.format(args.model, time.strftime("%Y%m%d-%H%M%S"))
    if not args.log_dir:
        saveDir = os.path.join('../checkpoints', childPath)
    else:
        saveDir = os.path.join(saveDir, childPath)
    os.makedirs(saveDir, exist_ok=True)
    return saveDir


if __name__ == '__main__':
    logger = infoLogger(name="test")
    deviceSetting(logger)
