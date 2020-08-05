# -*- coding: utf-8 -*-
# Time    : 2020/8/5 11:40
# Author  : zlich
# Filename: runonnx.py
import onnx
import time
import onnxruntime
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def runOnnx(path, image):
    print(onnxruntime.get_device())
    # onnxruntime.devi
    session = onnxruntime.InferenceSession(path)
    # print("The model expects input shape: ", session.get_inputs()[0].shape)
    detect(session, image)
    # session.run()


def detect(session, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    plt.imshow(resized[:, :, ::-1])
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(input_name)
    print(output_name)
    for i in range(100):
        init_time = time.time()
        outputs = session.run([output_name], {input_name: img_in})
        out = outputs[0][0]
        out_np = np.argmax(out, axis=0)
        print(time.time() - init_time)
        # print(out_np.shape)
    plt.figure()
    plt.imshow(out_np)
    plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--onnxpath", type=str, default='onnx_1_3_360_640.onnx')
    parse.add_argument("--device", type=str, default='0')
    parse.add_argument("--source", type=str, default='/home/hezl/laneDatasets/Bdd100k/images/val/be5ca08c-52c0860e.jpg')
    # parse.add_argument("")

    args = parse.parse_args()
    image_src = cv2.imread(args.source)
    runOnnx(args.onnxpath, image_src)
