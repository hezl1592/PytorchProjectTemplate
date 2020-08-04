# -*- coding: utf-8 -*-
# Time    : 2020/8/4 11:04
# Author  : zlich
# Filename: vis_feature.py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import Deeplabv3plus_Mobilenet
import torch.nn.functional as F
from utils.test_utils import modelDeploy, cv2Tensor
from utils.train_utils import deviceSetting
import torch
import cv2
import torchvision


class vismodel(Deeplabv3plus_Mobilenet):
    def __init__(self, output_channels=19, output_stride=8):
        super(vismodel, self).__init__(output_channels, output_stride)

    def forward(self, inputs):
        x, low_level_feat = self.encoder(inputs)
        self.feature1, self.feature2 = x.detach(), low_level_feat.detach()
        x = self.spp(x)
        x = self.decoder(x, low_level_feat)
        x = self.logits(x)

        return F.interpolate(x, size=inputs.shape[2:], mode='bilinear', align_corners=True)


def feature_imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.figure(figsize=(20, 20))
    inp = inp.detach().numpy().transpose((1, 2, 0))

    # mean = np.array([0.5, 0.5, 0.5])
    #
    # std = np.array([0.5, 0.5, 0.5])
    #
    # inp = std * inp + mean
    #
    # inp = np.clip(inp, 0, 1)

    plt.imshow(inp,cmap='jet')

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--modelpath", type=str,
                       default='../checkpoints/deeplabv3+_20200802-140613/deeplabv3+_640_360_best_37_0.791740.pth')
    parse.add_argument("--device", type=str, default='0')
    parse.add_argument("--source", type=str, default='/home/hezl/laneDatasets/Bdd100k/images/val/be5ca08c-52c0860e.jpg')
    # parse.add_argument("")

    args = parse.parse_args()
    _, args.device = deviceSetting(device=args.device)
    model = vismodel(3, 16)
    model = modelDeploy(args=args, model=model).to(args.device)
    model.eval()
    image = cv2.imread(args.source, -1)
    image, image_tensor = cv2Tensor(image, (640, 360))
    plt.figure()
    plt.imshow(image[:, :, ::-1])

    with torch.no_grad():
        # print(image_tensor.shape)
        pred = model(image_tensor.to(args.device))
        pred = pred.argmax(dim=1)
        # inference_time = time.time() - init_time
        pred_np = pred.cpu().detach().numpy()
        plt.figure()
        plt.imshow(pred_np[0])
        # out = torchvision.utils.make_grid(model.feature1[:, :20, :, :].transpose(1, 0).cpu(), nrow=5, normalize=True)
        # feature_imshow(out)
        #
        # out = torchvision.utils.make_grid(model.feature2[:, :20, :, :].transpose(1, 0).cpu(), nrow=5, normalize=True)
        # feature_imshow(out)
    plt.show()
