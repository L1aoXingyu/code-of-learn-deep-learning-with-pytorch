__author__ = 'SherlockLiao'

import os
import time

import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# define image transforms to do data augumentation
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val':
    transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# define data folder using ImageFolder to get images and classes from folder
root = '/media/sherlock/Files/kaggle_dog_vs_cat/'
data_folder = {
    'train':
    ImageFolder(
        os.path.join(root, 'data/train'), transform=data_transforms['train']),
    'val':
    ImageFolder(
        os.path.join(root, 'data/val'), transform=data_transforms['val'])
}

# define dataloader to load images
batch_size = 32
dataloader = {
    'train':
    DataLoader(
        data_folder['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4),
    'val':
    DataLoader(data_folder['val'], batch_size=batch_size, num_workers=4)
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
fix_param = True
# define model
transfer_model = models.resnet18(pretrained=True)
if fix_param:
    for param in transfer_model.parameters():
        param.requires_grad = False
dim_in = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(dim_in, 2)
if use_gpu:
    transfer_model = transfer_model.cuda()

# define optimize function and loss function
if fix_param:
    optimizer = optim.Adam(transfer_model.fc.parameters(), lr=1e-3)
else:
    optimizer = optim.Adam(transfer_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# train
num_epoch = 10

for epoch in range(num_epoch):
    print('{}/{}'.format(epoch + 1, num_epoch))
    print('*' * 10)
    print('Train')
    transfer_model.train()
    running_loss = 0.0
    running_acc = 0.0
    since = time.time()
    for i, data in enumerate(dataloader['train'], 1):
        img, label = data
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)

        # forward
        out = transfer_model(img)
        loss = criterion(out, label)
        _, pred = torch.max(out, 1)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * label.size(0)
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.data[0]
        if i % 100 == 0:
            print('Loss: {:.6f}, Acc: {:.4f}'.format(running_loss / (
                i * batch_size), running_acc / (i * batch_size)))
    running_loss /= data_size['train']
    running_acc /= data_size['train']
    elips_time = time.time() - since
    print('Loss: {:.6f}, Acc: {:.4f}, Time: {:.0f}s'.format(
        running_loss, running_acc, elips_time))
    print('Validation')
    transfer_model.eval()
    num_correct = 0.0
    total = 0.0
    eval_loss = 0.0
    for data in dataloader['val']:
        img, label = data
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
        out = transfer_model(img)
        _, pred = torch.max(out.data, 1)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        num_correct += (pred.cpu() == label.data.cpu()).sum()
        total += label.size(0)
    print('Loss: {:.6f} Acc: {:.4f}'.format(eval_loss / total, num_correct /
                                            total))
    print()
print('Finish Training!')
print()
save_path = os.path.join(root, 'model_save')
if not os.path.exists(save_path):
    os.mkdir(save_path)
torch.save(transfer_model.state_dict(), save_path + '/resnet18.pth')
