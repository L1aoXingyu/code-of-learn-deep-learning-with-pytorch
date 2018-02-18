# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import copy

import torch
from config import opt
from mxtorch import meter
from mxtorch import transforms as tfs
from mxtorch.trainer import *
from mxtorch.vision import model_zoo
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

train_tf = tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def test_tf(img):
    img = tfs.Resize(256)(img)
    img, _ = tfs.CenterCrop(224)(img)
    normalize = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = normalize(img)
    return img


def get_train_data():
    train_set = ImageFolder(opt.train_data_path, train_tf)
    return DataLoader(
        train_set, opt.batch_size, True, num_workers=opt.num_workers)


def get_test_data():
    test_set = ImageFolder(opt.test_data_path, test_tf)
    return DataLoader(
        test_set, opt.batch_size, True, num_workers=opt.num_workers)


def get_model():
    model = model_zoo.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    if opt.use_gpu:
        model = model.cuda(opt.ctx)
    return model


def get_loss(score, label):
    return nn.CrossEntropyLoss()(score, label)


def get_optimizer(model):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay)
    return ScheduledOptim(optimizer)


class FineTuneTrainer(Trainer):
    def __init__(self):
        model = get_model()
        criterion = get_loss
        optimizer = get_optimizer(model)
        super().__init__(model, criterion, optimizer)

        self.metric_meter['loss'] = meter.AverageValueMeter()
        self.metric_meter['acc'] = meter.AverageValueMeter()

    def train(self, kwargs):
        self.reset_meter()
        self.model.train()
        train_data = kwargs['train_data']
        for data in tqdm(train_data):
            img, label = data
            if opt.use_gpu:
                img = img.cuda(opt.ctx)
                label = label.cuda(opt.ctx)
            img = Variable(img)
            label = Variable(label)

            # Forward.
            score = self.model(img)
            loss = self.criterion(score, label)

            # Backward.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update meters.
            acc = (score.max(1)[1] == label).float().mean()
            self.metric_meter['loss'].add(loss.data[0])
            self.metric_meter['acc'].add(acc.data[0])

            # Update to tensorboard.
            if (self.n_iter + 1) % opt.plot_freq == 0:
                self.writer.add_scalars(
                    'loss', {'train': self.metric_meter['loss'].value()[0]},
                    self.n_plot)
                self.writer.add_scalars(
                    'acc', {'train': self.metric_meter['acc'].value()[0]},
                    self.n_plot)
                self.n_plot += 1
            self.n_iter += 1

        # Log the train metric dict to print result.
        self.metric_log['train loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['train acc'] = self.metric_meter['acc'].value()[0]

    def test(self, kwargs):
        self.reset_meter()
        self.model.eval()
        test_data = kwargs['test_data']
        for data in tqdm(test_data):
            img, label = data
            if opt.use_gpu:
                img = img.cuda(opt.ctx)
                label = label.cuda(opt.ctx)
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)

            score = self.model(img)
            loss = self.criterion(score, label)
            acc = (score.max(1)[1] == label).float().mean()

            self.metric_meter['loss'].add(loss.data[0])
            self.metric_meter['acc'].add(acc.data[0])

        # Update to tensorboard.
        self.writer.add_scalars('loss',
                                {'test': self.metric_meter['loss'].value()[0]},
                                self.n_plot)
        self.writer.add_scalars(
            'acc', {'test': self.metric_meter['acc'].value()[0]}, self.n_plot)
        self.n_plot += 1

        # Log the test metric to dict.
        self.metric_log['test loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['test acc'] = self.metric_meter['acc'].value()[0]

    def get_best_model(self):
        if self.metric_log['test loss'] < self.best_metric:
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.best_metric = self.metric_log['test loss']


def train(**kwargs):
    opt._parse(kwargs)

    train_data = get_train_data()
    test_data = get_test_data()

    fine_tune_trainer = FineTuneTrainer()
    fine_tune_trainer.fit(train_data=train_data, test_data=test_data)


if __name__ == '__main__':
    import fire

    fire.Fire()
