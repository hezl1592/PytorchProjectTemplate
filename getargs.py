# -*- coding: utf-8 -*-
# Time    : 2020/7/15 14:04
# Author  : zlich
# Filename: getargs.py
import argparse
import yaml
import sys


def getArgs(argv):
    sys.argv = argv
    # print(sys.argv)
    parser = argparse.ArgumentParser()

    # main
    parser.add_argument("--model", default="deeplabv3+", type=str, help="model name")

    # dir
    parser.add_argument("--log_dir", default="../checkpoints", type=str, help='path to checkpoint to store')

    # train
    parser.add_argument("--batch_size", default=16, type=int, help='train batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    parser.add_argument('--resume', default=None, type=str, help='path to checkpoint to resume from')
    parser.add_argument('--freeze_bn', default=False, action='store_true', help='Freeze BN params or not')

    args = parser.parse_args()
    # for name, value in vars(args).items():
    #     print(name, value)
    return args
