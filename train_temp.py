# -*- coding: utf-8 -*-
# Time    : 2020/7/15 13:07
# Author  : zlich
# Filename: train_temp.py
import torch
# from utils.print_utils import *
import os
from utils.train_utils import deviceSetting
from utils.log_utils import infoLogger

from model.net import parsingNet
from getargs import getArgs
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


def main(argv):
    # arguments
    args = getArgs(argv=argv)
    logger = infoLogger(log_dir='./log', name="train")
    logger.warning("xxxxxxxxxxxxxx")
    logger.debug("ccccccccc")
    logger.info("vvvvvvvvvvv")
    # print(vars(args))

    # device
    gpuNum, device = deviceSetting(logger=logger)
    # model
    model = parsingNet()


if __name__ == '__main__':
    main(sys.argv)

    # print(args)
