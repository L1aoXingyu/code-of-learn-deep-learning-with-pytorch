import torch
import torch.nn as nn
import torchvision.models as models

import loss

vgg = models.vgg19(pretrained=True).features
if torch.cuda.is_available():
    vgg = vgg.cuda()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_loss(style_img,
                             content_img,
                             cnn=vgg,
                             style_weight=1000,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):

    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential()
    if torch.cuda.is_available():
        model = model.cuda()
    gram = loss.Gram()
    if torch.cuda.is_available():
        gram = gram.cuda()

    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            if name in content_layers_default:
                target = model(content_img)
                content_loss = loss.Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                style_loss = loss.Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    return model, style_loss_list, content_loss_list
