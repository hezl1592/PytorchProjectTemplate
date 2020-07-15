# -*- coding: utf-8 -*-
# Time    : 2020/7/15 13:32
# Author  : zlich
# Filename: train_utils.py
import torch
# from utils.print_utils import *
from utils.log_utils import infoLogger

def deviceSetting(logger):
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    if num_gpus >= 1:
        logger.info("GPU found, device: {}, number: {}.".format(device, num_gpus))
    else:
        logger.warning("No GPU found, device: CPU.")
        # print_warning_message()
    return num_gpus, device


if __name__ == '__main__':
    logger = infoLogger(name="test")
    deviceSetting(logger)
