# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import os

import numpy as np
import torch
from PIL import Image
from mxtorch import transforms as tfs


def read_images(root, train):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/') + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


def random_crop(data, label, crop_size):
    height, width = crop_size
    data, rect = tfs.RandomCrop((height, width))(data)
    label = tfs.FixedCrop(*rect)(label)
    return data, label


def image2label(img):
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(COLORMAP):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
    return np.array(cm2lbl[idx], dtype=np.int64)


def img_transforms(img, label, crop_size):
    img, label = random_crop(img, label, crop_size)
    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = img_tfs(img)
    label = image2label(label)
    label = torch.from_numpy(label)
    return img, label


def inverse_normalization(img):
    """Convert normalized image to origin image.

    :param img:(~torch.FloatTensor) normalized image, (C, H, W)
    :return:
        Origin image.
    """
    img = img * torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None] \
          + torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]
    origin_img = torch.clamp(img, min=0, max=1) * 255
    origin_img = origin_img.permute(1, 2, 0).numpy()
    return origin_img.astype(np.uint8)


class VocSegDataset(object):
    def __init__(self, voc_root, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(voc_root, train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

    def _filter(self, images):
        return [img for img in images if (Image.open(img).size[1] >= self.crop_size[0] and
                                          Image.open(img).size[0] >= self.crop_size[1])]

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)


CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
