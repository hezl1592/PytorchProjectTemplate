# -*- coding: utf-8 -*-
# Time    : 2020/7/14 16:43
# Author  : zlich
# Filename: log_utils.py
from logging import getLogger, StreamHandler, INFO, DEBUG, Formatter, FileHandler
import os


# class Formatter_(Formatter):
#     def __int__(self, fmt=None):
#         super(Formatter_, self).__int__(fmt=fmt)
#     default_time_format = '%Y-%m-%d %H:%M:%S'
#     default_msec_format = '%s.%03d'

def infoLogger(log_dir='log', name='Model'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = getLogger(name)
    logger.setLevel(DEBUG)
    fmt_print = Formatter('%(asctime)s | %(name)s | \033[34m\033[1m-%(levelname)-7s- \033[0m |  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    fmt_log = Formatter('%(asctime)s | %(name)s | -%(levelname)-7s- |  %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    sh.setFormatter(fmt_print)
    logger.addHandler(sh)

    # fh = FileHandler(filename=log_dir.joinpath('debug.txt'), mode='w')
    debugFile = os.path.join(log_dir, 'logInfo.log')
    fh = FileHandler(debugFile, mode='w')
    fh.setLevel(DEBUG)
    fh.setFormatter(fmt_log)
    logger.addHandler(fh)
    return logger


if __name__ == '__main__':
    log = infoLogger()
    log.debug("...........")
    log.info("xxxxxxxxxx")
    log.warning("xxxxxxxxxx")
