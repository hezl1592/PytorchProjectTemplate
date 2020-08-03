import torch
import time
import numpy
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Deeplabv3plus_Mobilenet
from utils.train_utils import deviceSetting, savePath
from utils.log_utils import infoLogger
from utils.test_utils import modelDeploy, ImageGet
from utils.optimizer import create_optimizer_
from losses.multi import MultiClassCriterion
from data.bdd100k_drivablearea import BDD100K_Area_Seg
from getargs import getArgs_, cfgInfo
import sys


def main(argv, configPath):
    args = getArgs_(argv, configPath)
    args.logdir = savePath(args)
    args.num_gpus, args.device = deviceSetting(device=args.device)
    # model
    model = Deeplabv3plus_Mobilenet(args.output_channels, output_stride=args.output_stride)
    model = modelDeploy(args=args, model=model).to(args.device)
    model.eval()

    image_thread = ImageGet(args.source, args.logdir, size=args.size, queueLen=10)
    image_thread.start()

    num = 0
    with torch.no_grad():
        while True:
            if image_thread.readQueue.empty() and image_thread.finish_signal:
                break
            if not image_thread.readQueue.empty():
                num += 1
                image_source, image_tensor = image_thread.readQueue.get()
                st = time.time()
                image_tensor = image_tensor.to(args.device)

                init_time = time.time()
                pred = model(image_tensor)
                pred = pred.argmax(dim=1)
                inference_time = time.time() - init_time
                pred_np = pred.cpu().detach().numpy()

                image_thread.saveQueue.put((image_source, pred_np))
                includeGPU = time.time() - st

                print("| {:3d} | inference: {:.2f}ms\t| include GPU: {:.2f}ms\t|".format(num, inference_time * 1000,
                                                                                         includeGPU * 1000))
                # if num >= 100:
                #     break
    image_thread.finish_signal = 1
    image_thread.join()


if __name__ == '__main__':
    configPath = 'config/deeplabv3p_mobilenetv2_test.yaml'
    main(sys.argv, configPath)
