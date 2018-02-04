# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import warnings
from pprint import pprint


class DefaultConfig(object):
    model = 'resnet50'
    # Dataset.
    train_data_path = './hymenoptera_data/train/'
    test_data_path = './hymenoptera_data/val/'

    # Store result and save models.
    # result_file = 'result.txt'
    save_file = './checkpoints/'
    save_freq = 30  # save model every N epochs
    save_best = True  # If save best test metric model.

    # Visualization results on tensorboard.
    # vis_dir = './vis/'
    plot_freq = 100  # plot in tensorboard every N iterations

    # Model hyperparameters.
    use_gpu = True  # use GPU or not
    ctx = 0  # running on which cuda device
    batch_size = 64  # batch size
    num_workers = 4  # how many workers for loading data
    max_epoch = 30
    lr = 1e-2  # initial learning rate
    momentum = 0
    weight_decay = 1e-4
    lr_decay = 0.95
    # lr_decay_freq = 10

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('=========user config==========')
        pprint(self._state_dict())
        print('============end===============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
