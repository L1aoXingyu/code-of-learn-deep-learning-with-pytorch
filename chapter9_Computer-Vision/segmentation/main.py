# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from mxtorch import meter
from mxtorch.trainer import Trainer, ScheduledOptim
from mxtorch.vision.eval_tools import eval_semantic_segmentation
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from config import opt
from data import VocSegDataset, img_transforms, COLORMAP, inverse_normalization

warnings.filterwarnings('ignore')

cm = np.array(COLORMAP, dtype=np.uint8)


def get_data(is_train):
    voc_data = VocSegDataset(opt.voc_root, is_train, opt.crop_size,
                             img_transforms)
    return DataLoader(
        voc_data, opt.batch_size, True, num_workers=opt.num_workers)


def get_model(num_classes):
    model = getattr(models, opt.model)(num_classes)
    if opt.use_gpu:
        model.cuda()
    return model


def get_optimizer(model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    return ScheduledOptim(optimizer)


def get_loss(scores, labels):
    scores = F.log_softmax(scores, dim=1)
    return torch.nn.NLLLoss2d()(scores, labels)


all_metrcis = ['loss', 'acc', 'iou']


class FcnTrainer(Trainer):
    def __init__(self):
        model = get_model(opt.num_classes)
        criterion = get_loss
        optimizer = get_optimizer(model)

        super().__init__(model=model, criterion=criterion, optimizer=optimizer)

        self.config += ('Crop size: ' + str(opt.crop_size) + '\n')
        self.best_metric = 0
        for m in all_metrcis:
            self.metric_meter[m] = meter.AverageValueMeter()

    def train(self, kwargs):
        self.reset_meter()
        self.model.train()
        train_data = kwargs['train_data']
        for data in tqdm(train_data):
            imgs, labels = data
            if opt.use_gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
            imgs = Variable(imgs)
            labels = Variable(labels)

            # Forward.
            scores = self.model(imgs)
            loss = self.criterion(scores, labels)

            # Backward.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update to metrics.
            pred_labels = scores.max(dim=1)[1].data.cpu().numpy()
            pred_labels = [i for i in pred_labels]

            true_labels = labels.data.cpu().numpy()
            true_labels = [i for i in true_labels]

            eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)
            self.metric_meter['loss'].add(loss.data[0])
            self.metric_meter['acc'].add(eval_metrics['mean_class_accuracy'])
            self.metric_meter['iou'].add(eval_metrics['miou'])

            if (self.n_iter + 1) % opt.plot_freq == 0:
                # Plot metrics curve in tensorboard.
                self.writer.add_scalars(
                    'loss', {'train': self.metric_meter['loss'].value()[0]},
                    self.n_plot)
                self.writer.add_scalars(
                    'acc', {'train': self.metric_meter['acc'].value()[0]},
                    self.n_plot)
                self.writer.add_scalars(
                    'iou', {'train': self.metric_meter['iou'].value()[0]},
                    self.n_plot)

                # Show segmentation images.
                # Get prediction segmentation and ground truth segmentation.
                origin_image = inverse_normalization(imgs[0].cpu().data)
                pred_seg = cm[pred_labels[0]]
                gt_seg = cm[true_labels[0]]

                self.writer.add_image('train ori_img', origin_image,
                                      self.n_plot)
                self.writer.add_image('train gt', gt_seg, self.n_plot)
                self.writer.add_image('train pred', pred_seg, self.n_plot)
                self.n_plot += 1

            self.n_iter += 1

        self.metric_log['Train Loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['Train Mean Class Accuracy'] = self.metric_meter[
            'acc'].value()[0]
        self.metric_log['Train Mean IoU'] = self.metric_meter['iou'].value()[0]

    def test(self, kwargs):
        self.reset_meter()
        self.model.eval()
        test_data = kwargs['test_data']
        for data in tqdm(test_data):
            imgs, labels = data
            if opt.use_gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
            imgs = Variable(imgs, volatile=True)
            labels = Variable(labels, volatile=True)

            # Forward.
            scores = self.model(imgs)
            loss = self.criterion(scores, labels)

            # Update to metrics.
            pred_labels = scores.max(dim=1)[1].data.cpu().numpy()
            pred_labels = [i for i in pred_labels]

            true_labels = labels.data.cpu().numpy()
            true_labels = [i for i in true_labels]

            eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)
            self.metric_meter['loss'].add(loss.data[0])
            self.metric_meter['acc'].add(eval_metrics['mean_class_accuracy'])
            self.metric_meter['iou'].add(eval_metrics['miou'])

        # Plot metrics curve in tensorboard.
        self.writer.add_scalars('loss',
                                {'test': self.metric_meter['loss'].value()[0]},
                                self.n_plot)
        self.writer.add_scalars(
            'acc', {'test': self.metric_meter['acc'].value()[0]}, self.n_plot)
        self.writer.add_scalars(
            'iou', {'test': self.metric_meter['iou'].value()[0]}, self.n_plot)

        origin_img = inverse_normalization(imgs[0].cpu().data)
        pred_seg = cm[pred_labels[0]]
        gt_seg = cm[true_labels[0]]
        self.writer.add_image('test ori_img', origin_img, self.n_plot)
        self.writer.add_image('test gt', gt_seg, self.n_plot)
        self.writer.add_image('test pred', pred_seg, self.n_plot)

        self.n_plot += 1

        self.metric_log['Test Loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['Test Mean Class Accuracy'] = self.metric_meter[
            'acc'].value()[0]
        self.metric_log['Test Mean IoU'] = self.metric_meter['iou'].value()[0]

    def get_best_model(self):
        if self.metric_log['Test Mean IoU'] > self.best_metric:
            self.best_model = deepcopy(self.model.state_dict())
            self.best_metric = self.metric_log['Test Mean IoU']


def train(**kwargs):
    opt._parse(kwargs)

    # Set default cuda device.
    torch.cuda.set_device(opt.ctx)

    fcn_trainer = FcnTrainer()
    train_data = get_data(is_train=True)
    test_data = get_data(is_train=False)
    fcn_trainer.fit(
        train_data=train_data, test_data=test_data, epochs=opt.max_epoch)


if __name__ == '__main__':
    import fire

    fire.Fire()
