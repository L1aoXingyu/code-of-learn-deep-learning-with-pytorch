__author__ = 'SherlockLiao'

import os
from tqdm import tqdm
import h5py
import numpy as np
import argparse

import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import feature_net, classifier

parse = argparse.ArgumentParser()
parse.add_argument(
    '--model', required=True, help='vgg, inceptionv3, resnet152')
parse.add_argument('--bs', type=int, default=32)
parse.add_argument('--phase', required=True, help='train, val')
opt = parse.parse_args()
print(opt)

img_transform = transforms.Compose([
    transforms.Scale(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = '/media/sherlock/Files/kaggle_dog_vs_cat/data'
data_folder = {
    'train': ImageFolder(os.path.join(root, 'train'), transform=img_transform),
    'val': ImageFolder(os.path.join(root, 'val'), transform=img_transform)
}

# define dataloader to load images
batch_size = opt.bs
dataloader = {
    'train':
    DataLoader(
        data_folder['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4),
    'val':
    DataLoader(
        data_folder['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
}

# get train data size and validation data size
data_size = {
    'train': len(dataloader['train'].dataset),
    'val': len(dataloader['val'].dataset)
}

# get numbers of classes
img_classes = len(dataloader['train'].dataset.classes)

# test if using GPU
use_gpu = torch.cuda.is_available()


def CreateFeature(model, phase, outputPath='.'):
    """
    Create h5py dataset for feature extraction.

    ARGS:
        outputPath    : h5py output path
        model         : used model
        labelList     : list of corresponding groundtruth texts
    """
    featurenet = feature_net(model)
    if use_gpu:
        featurenet.cuda()
    feature_map = torch.FloatTensor()
    label_map = torch.LongTensor()
    for data in tqdm(dataloader[phase]):
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
        out = featurenet(img)
        feature_map = torch.cat((feature_map, out.cpu().data), 0)
        label_map = torch.cat((label_map, label), 0)
    feature_map = feature_map.numpy()
    label_map = label_map.numpy()
    file_name = '_feature_{}.hd5f'.format(model)
    h5_path = os.path.join(outputPath, phase) + file_name
    with h5py.File(h5_path, 'w') as h:
        h.create_dataset('data', data=feature_map)
        h.create_dataset('label', data=label_map)


CreateFeature(opt.model, opt.phase)
