# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
from copy import deepcopy

import numpy as np
import torch
from mxtorch import meter
from mxtorch.trainer import Trainer, ScheduledOptim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from config import opt
from data import TextDataset, TextConverter


def get_data(convert):
    dataset = TextDataset(opt.txt, opt.len, convert.text_to_arr)
    return DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers)


def get_model(convert):
    model = getattr(models, opt.model)(convert.vocab_size,
                                       opt.embed_dim,
                                       opt.hidden_size,
                                       opt.num_layers,
                                       opt.dropout)
    if opt.use_gpu:
        model = model.cuda()
    return model


def get_loss(score, label):
    return nn.CrossEntropyLoss()(score, label.view(-1))


def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    return ScheduledOptim(optimizer)


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


class CharRNNTrainer(Trainer):
    def __init__(self, convert):
        self.convert = convert

        model = get_model(convert)
        criterion = get_loss
        optimizer = get_optimizer(model)
        super().__init__(model, criterion, optimizer)
        self.config += ('text: ' + opt.txt + '\n' + 'train text length: ' + str(opt.len) + '\n')
        self.config += ('predict text length: ' + str(opt.predict_len) + '\n')

        self.metric_meter['loss'] = meter.AverageValueMeter()

    def train(self, kwargs):
        self.reset_meter()
        self.model.train()
        train_data = kwargs['train_data']
        for data in tqdm(train_data):
            x, y = data
            y = y.long()
            if opt.use_gpu:
                x = x.cuda()
                y = y.cuda()
            x, y = Variable(x), Variable(y)

            # Forward.
            score, _ = self.model(x)
            loss = self.criterion(score, y)

            # Backward.
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient.
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            self.optimizer.step()

            self.metric_meter['loss'].add(loss.data[0])

            # Update to tensorboard.
            if (self.n_iter + 1) % opt.plot_freq == 0:
                self.writer.add_scalar('perplexity', np.exp(self.metric_meter['loss'].value()[0]), self.n_plot)
                self.n_plot += 1

            self.n_iter += 1

        # Log the train metrics to dict.
        self.metric_log['perplexity'] = np.exp(self.metric_meter['loss'].value()[0])

    def test(self, kwargs):
        """Set beginning words and predicted length, using model to generate texts.

        Returns:
            predicted generating text
        """
        self.model.eval()
        begin = np.array([i for i in kwargs['begin']])
        begin = np.random.choice(begin, size=1)
        text_len = kwargs['predict_len']
        samples = [self.convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]
        if opt.use_gpu:
            input_txt = input_txt.cuda()
        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(text_len):
            out, init_state = self.model(model_input, init_state)
            pred = pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(pred))[None]
            if opt.use_gpu:
                model_input = model_input.cuda()
            result.append(pred[0])

        # Update generating txt to tensorboard.
        self.writer.add_text('text', self.convert.arr_to_text(result), self.n_plot)
        self.n_plot += 1
        print(self.convert.arr_to_text(result))

    def predict(self, begin, predict_len):
        self.model.eval()
        samples = [self.convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]
        if opt.use_gpu:
            input_txt = input_txt.cuda()
        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(predict_len):
            out, init_state = self.model(model_input, init_state)
            pred = pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(pred))[None]
            if opt.use_gpu:
                model_input = model_input.cuda()
            result.append(pred[0])
        text = self.convert.arr_to_text(result)
        print('Generate text is: {}'.format(text))
        with open(opt.write_file, 'a') as f:
            f.write(text)

    def load_state_dict(self, checkpoints):
        self.model.load_state_dict(torch.load(checkpoints))

    def get_best_model(self):
        if self.metric_log['perplexity'] < self.best_metric:
            self.best_model = deepcopy(self.model.state_dict())
            self.best_metric = self.metric_log['perplexity']


def train(**kwargs):
    opt._parse(kwargs)
    torch.cuda.set_device(opt.ctx)
    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    train_data = get_data(convert)
    char_rnn_trainer = CharRNNTrainer(convert)
    char_rnn_trainer.fit(train_data=train_data,
                         epochs=opt.max_epoch,
                         begin=opt.begin,
                         predict_len=opt.predict_len)


def predict(**kwargs):
    opt._parse(kwargs)
    torch.cuda.set_device(opt.ctx)
    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    char_rnn_trainer = CharRNNTrainer(convert)
    char_rnn_trainer.load_state_dict(opt.load_model)
    char_rnn_trainer.predict(opt.begin, opt.predict_len)


if __name__ == '__main__':
    import fire

    fire.Fire()
