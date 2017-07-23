__author__ = 'SherlockLiao'

import torch
from torchvision import models
from torch import nn


class feature_net(nn.Module):
    def __init__(self, model):
        super(feature_net, self).__init__()

        if model == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average', nn.AvgPool2d(9))
        elif model == 'inceptionv3':
            inception = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_module('global average', nn.AvgPool2d(35))
        elif model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """
        model includes vgg19, inceptionv3, resnet152
        """
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x


class classifier(nn.Module):
    def __init__(self, dim, n_classes):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, n_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
