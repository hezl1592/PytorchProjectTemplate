from functools import partial
import numpy as np
from pathlib import Path
import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class BDD100K_Area_Seg(Dataset):
    n_classes = 3
    void_classes = []
    valid_classes = [0, 1, 2]
    class_map = dict(zip(valid_classes, range(n_classes)))

    def __init__(self, base_dir='./dataset/bdd100k', split='train', target_size=(1280, 720), dataLen=None,
                 net_type='deeplab', ignore_index=255, debug=False,
                 affine_augmenter=None, image_augmenter=None):
        self.debug = debug
        assert os.path.exists(base_dir), "{} is not exists, check.".format(base_dir)
        self.base_dir = Path(base_dir)
        assert net_type in ['unet', 'deeplab']
        self.net_type = net_type
        self.ignore_index = ignore_index
        self.split = 'val' if split == 'valid' else split

        self.img_paths = sorted(self.base_dir.glob(f'images/{self.split}/*.jpg'))[:dataLen]
        self.lbl_paths = sorted(self.base_dir.glob(f'labels/{self.split}/*drivable_id.png'))[:dataLen]
        assert len(self.img_paths) == len(self.lbl_paths), "image num erro"

        # Resize

        # print(type(self.size), self.size)
        if isinstance(target_size, str):
            target_size = eval(target_size)
        self.size = target_size

        # Augment
        if self.split == 'train':
            self.affine_augmenter = affine_augmenter
            self.image_augmenter = image_augmenter
        else:
            self.affine_augmenter = None
            self.image_augmenter = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image_source = cv2.imread(str(img_path), -1)
        image_source = cv2.resize(image_source, self.size, cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image_source.transpose((2, 0, 1)))
        image_tensor = image_tensor.float().div(255)
        image_tensor = image_tensor.sub(0.5).div(0.5)

        lbl_path = self.lbl_paths[index]
        label_source = cv2.imread(str(lbl_path), -1)
        label_source = cv2.resize(label_source, self.size, cv2.INTER_NEAREST)
        lbl = self.encode_mask(label_source)
        label_tensor = torch.LongTensor(lbl)

        return image_tensor, label_tensor

    def encode_mask(self, lbl):
        for c in self.void_classes:
            lbl[lbl == c] = self.ignore_index
        for c in self.valid_classes:
            lbl[lbl == c] = self.class_map[c]
        return lbl
