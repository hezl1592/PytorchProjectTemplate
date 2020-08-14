# -*- coding: utf-8 -*-
# Time    : 2020/7/15 13:07
# Author  : zlich
# Filename: train_temp.py
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Deeplabv3plus_Mobilenet
from utils.train_utils import deviceSetting, savePath, modelDeploy
from utils.log_utils import infoLogger
from utils.train_utils import train_seg, val_seg, save_checkpoint
from utils.optimizer import create_optimizer_
from losses.multi import MultiClassCriterion
from data.bdd100k_drivablearea import BDD100K_Area_Seg
from getargs import getArgs_, cfgInfo
import numpy as np
import sys
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'


def main(argv, configPath=None):
    # arguments
    args = getArgs_(argv, configPath)
    saveDir = savePath(args)
    logger = infoLogger(logdir=saveDir, name=args.model)

    logger.info(argv)
    logger.debug(cfgInfo(args))
    logger.info("CheckPoints path: {}".format(saveDir))
    logger.debug("Model Name: {}".format(args.model))

    train_dataset = BDD100K_Area_Seg(base_dir=args.dataPath, split='train', target_size=args.size)
    valid_dataset = BDD100K_Area_Seg(base_dir=args.dataPath, split='val', target_size=args.size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_worker, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_worker, pin_memory=True)

    args.num_gpus, args.device = deviceSetting(logger=logger, device=args.device)
    # model
    model = Deeplabv3plus_Mobilenet(args.output_channels, output_stride=args.output_stride)

    optimizer, scheduler = create_optimizer_(model, args)
    loss_fn = MultiClassCriterion(loss_type=args.loss_type, ignore_index=args.ignore_index)
    model, trainData = modelDeploy(args, model, optimizer, scheduler, logger)

    tensorLogger = SummaryWriter(log_dir=os.path.join(saveDir, 'runs'), filename_suffix=args.model)
    logger.info("Tensorboard event log saved in {}".format(tensorLogger.log_dir))

    logger.info('Start training...')
    # global_step = 0
    start_epoch = trainData['epoch']

    num_classes = args.output_channels
    extra_info_ckpt = '{}_{}_{}'.format(args.model, args.size[0], args.size[1])
    for i_epoch in range(start_epoch, args.max_epoch):
        if i_epoch >= 29:
            optimizer.param_groups[0]["lr"] = np.float64(0.00001)
        trainData['epoch'] = i_epoch
        lossList, miouList = train_seg(model, train_loader, i_epoch, optimizer, loss_fn,
                                       num_classes, logger, tensorLogger, args=args)
        scheduler.step()
        trainData['loss'].extend(lossList)
        trainData['miou'].extend(miouList)

        valLoss, valMiou = val_seg(model, valid_loader, i_epoch, loss_fn,
                                   num_classes, logger, tensorLogger, args=args)
        trainData['val'].append([valLoss, valMiou])

        best = valMiou > trainData['bestMiou']
        if valMiou > trainData['bestMiou']:
            trainData['bestMiou'] = valMiou

        weights_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()

        save_checkpoint({'trainData': trainData,
                         'model': weights_dict,
                         'optimizer': optimizer.state_dict(),
                         }, is_best=best, dir=saveDir, extra_info=extra_info_ckpt, miou_val=valMiou, logger=logger)

    tensorLogger.close()


if __name__ == '__main__':
    configPath = 'config/deeplabv3p_mobilenetv2.yaml'
    main(sys.argv, configPath)

    # print(args)
