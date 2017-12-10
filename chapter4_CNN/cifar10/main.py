import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Image Preprocessing
train_transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(
    root='./data', train=True, transform=train_transform, download=True)

test_dataset = dsets.CIFAR10(
    root='./data', train=False, transform=test_transform)

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


resnet = ResNet(ResidualBlock, [3, 3, 3])
resnet.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

# Training
total_epoch = 50
for epoch in range(total_epoch):
    running_loss = 0
    running_acc = 0
    running_num = 0
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # =====================log=====================
        running_num += labels.size(0)
        running_loss += loss.data[0] * labels.size(0)
        _, correct_label = torch.max(outputs, 1)
        correct_num = (correct_label == labels).sum()
        running_acc += correct_num.data[0]
        if (i + 1) % 100 == 0:
            print_loss = running_loss / running_num
            print_acc = running_acc / running_num
            print("Epoch [{}/{}], Iter [{}/{}] Loss: {:.6f} Acc: {:.6f}".
                  format(epoch + 1, total_epoch, i + 1,
                         len(train_loader), print_loss, print_acc))

    # Decaying Learning Rate
    if (epoch + 1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

# Test
correct = 0
total = 0
for images, labels in test_loader:
    if torch.cuda.is_available:
        images = Variable(images.cuda())
    else:
        images = Variable(images)
    outputs = resnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the model on the test images: {:.2f} %%'.format(
    100 * correct / total))

# Save the Model
torch.save(resnet.state_dict(), 'resnet.pth')
