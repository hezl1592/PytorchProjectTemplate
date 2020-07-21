# -*- coding: utf-8 -*-
# Time    : 2020/7/15 13:07
# Author  : zlich
# Filename: train_temp.py
import torch
# from utils.print_utils import *
import os
from utils.train_utils import deviceSetting, savePath
from utils.log_utils import infoLogger
from utils.factory import get_scheduler, get_optimizer
from model.net import parsingNet
from getargs import getArgs
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


def main(argv):
    # arguments
    args = getArgs(argv=argv)
    saveDir = savePath(args)
    logger = infoLogger(log_dir=saveDir, name=args.model)
    logger.info("CheckPoints path: {}".format(saveDir))
    logger.debug("Model Name: {}".format(args.model))
    logger.warning("Model Name: {}".format(args.model))
    num_gpus, device = deviceSetting(logger=logger)
    # model
    model = parsingNet()

    # data
    trainLoader =

    # optimizer
    optimizer = get_optimizer(net=model, cfg=args)
    scheduler = get_scheduler(optimizer, cfg=args, len())



    if num_gpus >= 1:
        from torch.nn.parallel import DataParallel
        model = DataParallel(model)
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    if torch.backends.cudnn.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

    # if args.resume:


if __name__ == '__main__':
    main(sys.argv)

    # print(args)
