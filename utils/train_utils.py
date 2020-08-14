# -*- coding: utf-8 -*-
# Time    : 2020/7/15 13:32
# Author  : zlich
# Filename: train_utils.py
import torch
import torch.nn as nn
import os
import numpy as np
# from utils.print_utils import *
from utils.log_utils import infoLogger
from utils.metrics import AverageMeter, Evaluator
import time
import yaml
import os
from .print_utils import print_info_message


def deviceSetting(logger=None, device=None):
    if not device:
        pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    if num_gpus >= 1:
        if logger:
            logger.info("GPU found, device: {}, number: {}.".format(device, num_gpus))
        else:
            print_info_message("GPU found, device: {}, number: {}.".format(device, num_gpus))
    else:
        if logger:
            logger.warning("No GPU found, device: CPU.")
        else:
            print_info_message("No GPU found, device: CPU.")
        # print_warning_message()
    return num_gpus, torch.device(device)


def savePath(args):
    saveDir = args.logdir
    saveDir = os.path.abspath(saveDir)
    childPath = '{}_{}'.format(args.model, time.strftime("%Y%m%d-%H%M%S"))
    if not args.logdir:
        saveDir = os.path.join('../checkpoints', childPath)
    else:
        saveDir = os.path.join(saveDir, childPath)

    os.makedirs(saveDir, exist_ok=True)

    return saveDir


def readYAML(path):
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as fs:
            fs = fs.read()
        config = yaml.load(fs)
        return config
    else:
        return None


def modelDeploy(args, model, optimizer, scheduler, logger):
    if args.num_gpus >= 1:
        from torch.nn.parallel import DataParallel
        model = DataParallel(model)
        model = model.cuda()

    if torch.backends.cudnn.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

    trainData = {'epoch': 0,
                 'loss': [],
                 'miou': [],
                 'val': [],
                 'bestMiou': 0
                 }

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))

            # model&optimizer
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # stop point
            trainData = checkpoint['trainData']
            for i in range(trainData['epoch']):
                scheduler.step()
            # print(trainData)

            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, trainData['epoch']))

        else:
            logger.error("=> no checkpoint found at '{}'".format(args.resume))
            assert False, "=> no checkpoint found at '{}'".format(args.resume)

    if args.finetune:
        if os.path.isfile(args.finetune):
            logger.info("=> finetuning checkpoint '{}'".format(args.finetune))
            state_all = torch.load(args.finetune, map_location='cpu')['model']
            state_clip = {}  # only use backbone parameters
            # print(model.state_dict().keys())
            for k, v in state_all.items():
                state_clip[k] = v
            # print(state_clip.keys())
            model.load_state_dict(state_clip, strict=False)
        else:
            logger.warning("finetune is not a file.")
            pass

    if args.freeze_bn:
        logger.warning('Freezing batch normalization layers')
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model, trainData


def train_seg(model, dataLoader, epoch, optimizer, loss_fn, num_classes,
              logger, tensorLogger, device='cuda', args=None):
    model.train()
    logger.info("Train | [{:2d}/{}] | Lr: {}  |".format(epoch + 1, args.max_epoch, optimizer.param_groups[0]["lr"]))
    tensorLogger.add_scalar("Common/lr", optimizer.param_groups[0]["lr"], epoch)
    losses = AverageMeter()
    batch_time = AverageMeter()
    Miou = AverageMeter()

    evaluator = Evaluator(num_class=num_classes)
    evaluator.reset()

    lossList = []
    miouList = []
    for i, (inputs, target) in enumerate(dataLoader):
        inputs = inputs.to(device=device)
        target = target.to(device=device)

        initTime = time.time()
        output = model(inputs)

        loss = loss_fn(output, target)

        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # print(output_np.shape, target_np.shape)
        evaluator.add_batch(target_np, np.argmax(output_np, axis=1))
        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - initTime)
        if i % 20 == 0:
            miou, iou = evaluator.Mean_Intersection_over_Union()
            Miou.update(miou, 20)
            tensorLogger.add_scalar('train/loss', losses.avg, epoch * len(dataLoader) + i)
            tensorLogger.add_scalar('train/miou', miou, epoch * len(dataLoader) + i)
            lossList.append(losses.avg)
            miouList.append(miou)

        if i % 100 == 0:  # print after every 100 batches
            logger.info("Train | {:2d} | [{:4d}/{}] Infer:{:.2f}sec |  Loss:{:.4f}  |  Miou:{:4f}  |".
                        format(epoch + 1, i + 1, len(dataLoader), batch_time.avg, losses.avg, miou))
            evaluator.reset()

    return lossList, miouList


def val_seg(model, dataLoader, epoch, loss_fn, num_classes, logger, tensorLogger, device='cuda', args=None):
    model.eval()
    logger.info("Valid | [{:2d}/{}]".format(epoch + 1, args.max_epoch))
    losses = AverageMeter()
    batch_time = AverageMeter()

    evaluator = Evaluator(num_class=num_classes)
    evaluator.reset()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataLoader):
            inputs = inputs.to(device=device)
            target = target.to(device=device)

            initTime = time.time()
            output = model(inputs)

            loss = loss_fn(output, target)

            output_np = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            # print(output_np.shape, target_np.shape)
            evaluator.add_batch(target_np, np.argmax(output_np, axis=1))
            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - initTime)

            if i % 100 == 0:  # print after every 100 batches
                logger.info("Valid | {:2d} | [{:4d}/{}] Infer:{:.2f}sec |  Loss:{:.4f}  |  Miou:{:4f}  |".
                            format(epoch + 1, i + 1, len(dataLoader), batch_time.avg, losses.avg,
                                   evaluator.Mean_Intersection_over_Union()[0]))
    Totalmiou = evaluator.Mean_Intersection_over_Union()[0]
    tensorLogger.add_scalar('val/loss', losses.avg, epoch + 1)
    tensorLogger.add_scalar('val/miou', Totalmiou, epoch + 1)

    return losses.avg, Totalmiou


def save_checkpoint(state, is_best, dir, extra_info='model', epoch=-1, miou_val=0, logger=None):
    check_pt_file = dir + os.sep + str(extra_info) + '_checkpoint_{}_{:6f}.pth.tar'.format(
        state['trainData']['epoch'] + 1,
        miou_val)
    torch.save(state, check_pt_file)
    if is_best:
        torch.save(state['model'],
                   dir + os.sep + str(extra_info) + '_best_{}_{:6f}.pth'.format(state['trainData']['epoch'] + 1,
                                                                                miou_val))
    if epoch != -1:
        torch.save(state['model'], dir + os.sep + str(extra_info) + '_ep_' + str(epoch) + '.pth')
    if logger:
        logger.info('Train | {:2d} | Checkpoint: {}'.format(state['trainData']['epoch'] + 1, check_pt_file))
    else:
        print_info_message('Train | {:2d} | Checkpoint: {}'.format(state['trainData']['epoch'] + 1, check_pt_file))


if __name__ == '__main__':
    logger = infoLogger(name="test")
    deviceSetting(logger)

    config = readYAML("../config/bdd100k_deeplabv3p_mobilenetv2_apex.yaml")
    print(config)
