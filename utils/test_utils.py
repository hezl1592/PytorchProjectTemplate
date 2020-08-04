import os
import threading
import time
from queue import Queue
import numpy as np
import cv2
import torch

_mask = np.array([[0, 0, 0],  # black
                  [0, 0, 255],  # red in BGR color system
                  [255, 0, 0],  # blue in BGR color system
                  [0, 255, 255]])  # yellow in BGR color system


def modelDeploy(args, model):
    if args.modelpath and os.path.isfile(args.modelpath):
        params = torch.load(args.modelpath, map_location=torch.device('cpu'))
        state_clip = {}
        for k, v in params.items():
            new_k = k.replace('module.', '')
            state_clip[new_k] = v
        model.load_state_dict(state_clip, strict=True)
        del params
    else:
        print("no modelPath, please check!")
        exit(-1)

    return model


def cv2Tensor(image, size=(640,360)):
    image_source = cv2.resize(image, size)
    image_tensor = torch.from_numpy(image_source.transpose((2, 0, 1)))
    image_tensor = image_tensor.float().div(255)
    image_tensor = image_tensor.sub(0.5).div(0.5)
    image_tensor = image_tensor.unsqueeze(0)
    return image_source, image_tensor


class ImageGet(object):
    def __init__(self, source, output, size=(640, 360), queueLen=1, video=False):
        super(ImageGet, self).__init__()
        self.finish_signal = 0
        self.source = source
        self.size = size
        self.readQueue = Queue(queueLen)
        self.output = output
        self.video = video
        print("Reading images from ", self.source)
        if os.path.isfile(self.source):
            if self.source.endswith('.png') or self.source.endswith('.jpg'):
                self.readthread = threading.Thread(target=self._imageGet, args=(self.source, self.readQueue))
            elif self.source.endswith('.avi') or self.source.endswith('.mp4'):
                self.video = True
                self.readthread = threading.Thread(target=self._videoGet, args=(self.source, self.readQueue))
        elif os.path.isdir(self.source):
            self.readthread = threading.Thread(target=self._imageDirGet, args=(self.source, self.readQueue))
        else:
            print("please Check!")
            exit(-1)

        self.saveQueue = Queue()
        self.saveThread = threading.Thread(target=self._Imagesave, args=(self.output, self.saveQueue))

    def _videoGet(self, Path, read_queue):
        cap = cv2.VideoCapture(Path)

        while cap.isOpened():
            _, image_source = cap.read()
            try:
                while read_queue.full():
                    time.sleep(0.01)
                else:
                    image_source, image_tensor = cv2Tensor(image_source, self.size)
                    read_queue.put((image_source, image_tensor))
            except:
                print("video done.")
                break
            # if cv2.waitKey(30) & 0xFF == ord('q'):
            #     break
            if self.finish_signal == 1:
                break
        self.finish_signal = 1
        cap.release()
        print("done!")

    def _imageGet(self, Path, read_queue):
        image = cv2.imread(Path, -1)
        image_source, image_tensor = cv2Tensor(image, self.size)
        read_queue.put((image_source, image_tensor))
        self.finish_signal = 1

    def _imageDirGet(self, Path, read_queue):
        for file in os.listdir(Path):
            if self.finish_signal == 1:
                break

            if file.endswith('jpg') or file.endswith('png'):
                image = cv2.imread(os.path.join(Path, file), -1)
                try:
                    while read_queue.full():
                        time.sleep(0.01)
                    else:
                        image_source, image_tensor = cv2Tensor(image, self.size)
                        read_queue.put((image_source, image_tensor))
                except:
                    print("dir done.")
                    break
        self.finish_signal = 1
        print("done!")

    def _Imagesave(self, output, saveQueue):
        if self.video:
            videoName = '{}/vis_out_color_640_360_2__.avi'.format(output)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vout = cv2.VideoWriter(videoName, fourcc, 30.0, (640, 1080))
        num = 0
        while True:
            if saveQueue.empty() and self.finish_signal:
                break
            if not saveQueue.empty():
                num += 1
                try:
                    image_source, image_pred = saveQueue.get()
                    pred_lbl = np.array(_mask[image_pred[0]], np.uint8)
                    blend = np.bitwise_or(image_source, pred_lbl)
                    show_image = cv2.vconcat([image_source, pred_lbl, blend])
                    if self.video:
                        vout.write(show_image)
                    else:
                        cv2.imwrite("{}/{:05d}.jpg".format(output, num), show_image)
                except:
                    print("done")
                    break
        if self.video:
            vout.release()
        self.finish_signal = 1
        print("save done")

    def start(self):
        self.readthread.start()
        self.saveThread.start()

    def join(self):
        self.finish_signal = 1
        self.readthread.join()
        self.saveThread.join()
