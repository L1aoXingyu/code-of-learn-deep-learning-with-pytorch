__author__ = 'SherlockLiao'

import argparse
import time
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader

from dataset import h5Dataset
from net import classifier

parse = argparse.ArgumentParser()
parse.add_argument(
    '--model',
    nargs='+',
    help='inceptionv3, vgg, resnet152',
    default=['vgg', 'inceptionv3', 'resnet152'])
parse.add_argument('--batch_size', type=int, default=64)
parse.add_argument('--epoch', type=int, default=20)
parse.add_argument('--n_classes', default=2, type=int)
parse.add_argument('--num_workers', type=int, default=8)
opt = parse.parse_args()
print(opt)

root = '/media/sherlock/Files/kaggle_dog_vs_cat/'
train_list = ['train_feature_{}.hd5f'.format(i) for i in opt.model]
val_list = ['val_feature_{}.hd5f'.format(i) for i in opt.model]

dataset = {'train': h5Dataset(train_list), 'val': h5Dataset(val_list)}

datasize = {
    'train': dataset['train'].dataset.size(0),
    'val': dataset['val'].dataset.size(0)
}

batch_size = opt.batch_size
epoches = opt.epoch

dataloader = {
    'train':
    DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=opt.num_workers),
    'val':
    DataLoader(
        dataset['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
}

dimension = dataset['train'].dataset.size(1)

mynet = classifier(dimension, opt.n_classes)
mynet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mynet.parameters(), lr=1e-3)
# train
for epoch in range(epoches):
    print('{}'.format(epoch + 1))
    print('*' * 10)
    print('Train')
    mynet.train()
    since = time.time()

    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(dataloader['train'], 1):
        feature, label = data
        feature = Variable(feature).cuda()
        label = Variable(label).cuda()

        # forward
        out = mynet(feature)
        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.data[0]
        if i % 50 == 0:
            print('Loss: {:.6f}, Acc: {:.6f}'.format(running_loss / (
                i * batch_size), running_acc / (i * batch_size)))

    running_loss /= datasize['train']
    running_acc /= datasize['train']
    eplise_time = time.time() - since
    print('Loss: {:.6f}, Acc: {:.6f}, Time: {:.0f}s'.format(
        running_loss, running_acc, eplise_time))
    print('Validation')
    mynet.eval()
    num_correct = 0.0
    eval_loss = 0.0
    for data in dataloader['val']:
        feature, label = data
        feature = Variable(feature, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
        # forward
        out = mynet(feature)
        loss = criterion(out, label)

        _, pred = torch.max(out, 1)
        correct = torch.sum(pred == label)
        num_correct += correct.data[0]
        eval_loss += loss.data[0] * label.size(0)

    print('Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / datasize['val'],
                                             num_correct / datasize['val']))
print('Finish Training!')

save_path = os.path.join(root, 'model_save')
if not os.path.exists(save_path):
    os.mkdir(save_path)

torch.save(mynet.state_dict(), save_path + '/feature_model.pth')
