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


def timeit(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("%.3fms for {%s}" % ((time.time() - t0) * 1000, func.__name__), end=' ')
        return back

    return newFunc


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


def cv2Tensor(image, size=(640, 360)):
    image_source = cv2.resize(image, size)
    image_tensor = torch.from_numpy(image_source.transpose((2, 0, 1)))
    image_tensor = image_tensor.float().div(255)
    image_tensor = image_tensor.sub(0.5).div(0.5)
    image_tensor = image_tensor.unsqueeze(0)
    return image_source, image_tensor


class ImageGet(object):
    def __init__(self, sourcePath, outputPath, size=(640, 360), queueLen=1,
                 save_image=False, save_video=False, fps=25, videoSize=(640, 320), videoName='vis'):
        super(ImageGet, self).__init__()
        self.finish_signal = 0
        self.source = sourcePath
        self.output = outputPath
        self.size = size
        self.readQueue = Queue(queueLen)
        self.video = save_video
        self.image = save_image
        if self.video:
            self.fps = fps
            self.videoSize = videoSize
            self.videoName = videoName

        print("Reading images from ", self.source)
        if os.path.isfile(self.source):
            if self.source.endswith('.png') or self.source.endswith('.jpg'):
                self.video = False
                self.readThread = threading.Thread(target=self._imageGet, args=(self.source, self.readQueue))
            elif self.source.endswith('.avi') or self.source.endswith('.mp4'):
                self.readThread = threading.Thread(target=self._videoGet, args=(self.source, self.readQueue))
        elif os.path.isdir(self.source):
            self.readThread = threading.Thread(target=self._imageDirGet, args=(self.source, self.readQueue))
        else:
            print("please Check!")
            exit(-1)

        self.saveQueue = Queue(100)
        self.saveThread = threading.Thread(target=self._Imagesave, args=(self.output, self.saveQueue))

    def _videoGet(self, Path, read_queue):
        cap = cv2.VideoCapture(Path)
        num = 0
        while cap.isOpened():
            _, image_source = cap.read()
            try:
                while read_queue.full() and self.saveQueue.full():
                    time.sleep(0.01)
                else:
                    num += 1
                    name = "{:06d}.jpg".format(num)
                    image_source, image_tensor = cv2Tensor(image_source, self.size)
                    read_queue.put((name, image_source, image_tensor))
            except:
                print("video done.")
                break
            if self.finish_signal == 1:
                break
        self.finish_signal = 1
        cap.release()
        print("done!")

    def _imageGet(self, Path, read_queue):
        image = cv2.imread(Path, -1)
        imageName = Path.split('/')[-1]
        image_source, image_tensor = cv2Tensor(image, self.size)
        read_queue.put((imageName, image_source, image_tensor))
        self.finish_signal = 1

    def _imageDirGet(self, Path, read_queue):
        for file in sorted(os.listdir(Path)[:1000]):
            if self.finish_signal == 1:
                break

            if file.endswith('jpg') or file.endswith('png'):
                image = cv2.imread(os.path.join(Path, file), -1)
                try:
                    while read_queue.full():
                        time.sleep(0.01)
                    else:
                        image_source, image_tensor = cv2Tensor(image, self.size)
                        read_queue.put((file, image_source, image_tensor))
                except:
                    print("dir done.")
                    break
        self.finish_signal = 1
        print("done!")

    def _Imagesave(self, output, saveQueue):
        if self.video:
            videoName = '{}/{}.avi'.format(output, self.videoName)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vout = cv2.VideoWriter(videoName, fourcc, self.fps, self.videoSize)
        time.sleep(5)
        while True:
            if saveQueue.empty() and self.finish_signal:
                break
            if not saveQueue.empty():
                try:
                    image_name, image_source, image_pred = saveQueue.get()
                    pred_lbl = np.array(_mask[image_pred[0]], np.uint8)
                    blend = np.bitwise_or(image_source, pred_lbl)
                    show_image = cv2.vconcat([image_source, pred_lbl, blend])
                    if self.video:
                        video_image = cv2.resize(show_image, self.videoSize)

                    if self.video and self.image:
                        vout.write(video_image)
                        cv2.imwrite("{}/{}".format(output, image_name), show_image)
                    elif self.video:
                        vout.write(video_image)
                    elif self.image:
                        cv2.imwrite("{}/{}".format(output, image_name), show_image)
                    else:
                        pass
                except:
                    print("done")
                    assert False
                    break
        if self.video:
            vout.release()
        self.finish_signal = 1
        print("save done")

    def start(self):
        self.readThread.start()
        self.saveThread.start()

    def join(self):
        self.finish_signal = 1
        self.readThread.join()
        self.saveThread.join()
