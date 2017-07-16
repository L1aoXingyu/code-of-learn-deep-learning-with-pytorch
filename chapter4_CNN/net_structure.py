import torch.nn as nn
from torch.nn import init


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # b, 3, 32, 32
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))
        # b, 32, 32, 32
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b, 32, 16, 16
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
        # b, 64, 16, 16
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))  # b, 64, 8, 8
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        #b, 128, 8, 8
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))  #b, 128, 4, 4
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out


model = SimpleCNN()

new_model = nn.Sequential(*list(model.children())[:2])

conv_model = nn.Sequential()
for layer in model.named_modules():
    if 'conv' in layer[0]:
        conv_model.add_module(layer[0], layer[1])

for param in model.named_parameters():
    if 'conv' in param[0] and 'weight' in param[0]:
        init.normal(param[1].data)
        init.xavier_normal(param[1].data)
        init.kaiming_normal(param[1].data)
