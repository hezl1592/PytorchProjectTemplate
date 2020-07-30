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

def infoLogger(logdir='log', name='Model'):
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    logger = getLogger(name)
    logger.setLevel(DEBUG)
    fmt_log = Formatter('%(asctime)s | %(name)s | -%(levelname)-4s- |  %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    sh.setFormatter(fmt_log)
    logger.addHandler(sh)

    # fh = FileHandler(filename=logdir.joinpath('debug.txt'), mode='w')
    debugFile = os.path.join(logdir, 'logInfo.log')
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
