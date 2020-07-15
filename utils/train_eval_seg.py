# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from utilities.utils import AverageMeter
import time
from utils.metrics.segmentation_miou import MIOU
from utils.print_utils import *
from torch.nn.parallel import gather
import numpy as np
import cv2
import os 

def train_seg(model, dataset_loader, optimizer, criterion, num_classes, epoch, writer, device='cuda'):
    losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    end = time.time()
    model.train()
    # print("....", device)
    miou_class = MIOU(num_classes=num_classes)

    lossList = []
    miouList = []
    for i, (inputs, target) in enumerate(dataset_loader):
        inputs = inputs.to(device=device)
        target = target.to(device=device)

        # print("inputs:", type(inputs), type(target))
        outputs = model(inputs)
        # print(type(outputs), len(outputs))

        if device == 'cuda':
            loss = criterion(outputs, target).mean()
            # print("loss", loss)
            if isinstance(outputs, (list, tuple)):
                target_dev = outputs[0].device
                outputs = gather(outputs, target_device=target_dev)
        else:
            loss = criterion(outputs, target)

        inter, union = miou_class.get_iou(outputs, target)

        inter_meter.update(inter)
        union_meter.update(union)

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:  # print after every 100 batches
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                              (epoch, i, len(dataset_loader), batch_time.avg, losses.avg, miou))
            writer.add_scalar('Segmentation/stepLoss/train', losses.avg, epoch * len(dataset_loader) + i)
            writer.add_scalar('Segmentation/stepmIOU/train', miou, epoch * len(dataset_loader) + i)
            lossList.append(losses.avg)
            miouList.append(miou)
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100
    return miou, losses.avg, lossList, miouList


def val_seg(model, dataset_loader, epoch, criterion=None, num_classes=21, device='cuda'):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes)

    if criterion:
        losses = AverageMeter()

    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataset_loader):
            inputs = inputs.to(device=device)
            target = target.to(device=device)
            
            init_time = time.time()
            outputs = model(inputs)
            
            inferenTime = time.time() - init_time
            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)
                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion is not None else 0
                print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f\t\tInference:%.3fsec" %
                                  (epoch,i, len(dataset_loader), batch_time.avg, loss_, miou, inferenTime))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))
    if criterion:
        return miou, losses.avg
    else:
        return miou, 0


def val_seg_with_vis(model, dataset_loader, epoch, colorMask, savedir, criterion=None, num_classes=21, device='cuda'):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes)

    if criterion:
        losses = AverageMeter()

    savedir = os.path.join(savedir, "eporch_{:03d}".format(epoch))
    os.makedirs(savedir, exist_ok=True)
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataset_loader):
            inputs = inputs.to(device=device)
            target = target.to(device=device)

            outputs = model(inputs)
            # print(type(outputs[0]))
            # print(inputs.shape, target.shape)
            # print("output len:", len(outputs), outputs[0].shape)
            if i * inputs.size(0) < 150:
                for index, outBatch in enumerate(outputs):
                    # print(index, outBatch.dim())
                    if outBatch.dim() == 4:
                        for indexTensor in range(outBatch.size(0)):
                            indexTotal = index * outBatch.size(0) + indexTensor

                            image = inputs[indexTotal].div_(2).add_(0.5).div_(1/255)
                            image = image.cpu().detach().numpy().transpose((1, 2, 0))
                            image = image.astype(np.uint8)

                            realLabel = target[indexTotal]
                            realLabel = realLabel.cpu().detach().numpy()
                            realLabel = np.array(colorMask[realLabel], np.uint8)
                            print(image.shape, realLabel.shape)

                            # print(outBatch[indexTensor].shape)
                            pred = outBatch[indexTensor].argmax(dim=0)
                            pred_out = pred.cpu().detach().numpy()
                            # print(pred_out.shape)
                            pred_lbl = np.array(colorMask[pred_out], np.uint8)
                            out_img = cv2.vconcat([image, realLabel, pred_lbl])
                            cv2.imwrite("{}/{}_{}.jpg".format(savedir, i, indexTotal), out_img)
                        # print(image.shape, realLabel.shape, pred_lbl.shape)
                    # print(image.dtype, realLabel.dtype, pred_lbl.dtype)
                    elif outBatch.dim() == 3:
                        image = inputs[index].div_(2).add_(0.5).div_(1/255)
                        image = image.cpu().detach().numpy().transpose((1, 2, 0))
                        image = image.astype(np.uint8)

                        realLabel = target[index]
                        realLabel = realLabel.cpu().detach().numpy()
                        realLabel = np.array(colorMask[realLabel], np.uint8)
                        print(image.shape, realLabel.shape)

                        # print(outBatch[indexTensor].shape)
                        pred = outBatch.argmax(dim=0)
                        pred_out = pred.cpu().detach().numpy()
                        # print(pred_out.shape)
                        pred_lbl = np.array(colorMask[pred_out], np.uint8)
                        out_img = cv2.vconcat([image, realLabel, pred_lbl])
                        cv2.imwrite("{}/{}_{}.jpg".format(savedir, i, index), out_img)
                    else:
                        print("val vis something error!")
                        break
                # for index in range(inputs.size(0)):

            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)
                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion is not None else 0
                print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                                  (epoch,i, len(dataset_loader), batch_time.avg, loss_, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))
    if criterion:
        return miou, losses.avg
    else:
        return miou, 0