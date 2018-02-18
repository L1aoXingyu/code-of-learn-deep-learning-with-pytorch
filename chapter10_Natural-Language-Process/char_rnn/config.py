# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import warnings
from pprint import pprint


class DefaultConfig(object):
    model = 'CharRNN'

    # Dataset.
    txt = './dataset/poetry.txt'
    len = 20
    max_vocab = 8000
    begin = '天青色等烟雨'  # begin word of text
    predict_len = 50  # predict length

    # Store result and save models.
    result_file = 'result.txt'
    save_file = './checkpoints/'
    save_freq = 30  # save model every N epochs
    save_best = True

    # Predict mode and generate contexts
    load_model = './checkpoints/CharRNN_best_model.pth'
    write_file = './write_context.txt'

    # Visualization parameters.
    vis_dir = './vis/'
    plot_freq = 100  # plot in tensorboard every N iterations

    # Model parameters.
    embed_dim = 512
    hidden_size = 512
    num_layers = 2
    dropout = 0.5

    # Model hyperparameters.
    use_gpu = True  # use GPU or not
    ctx = 0  # running on which cuda device
    batch_size = 128  # batch size
    num_workers = 4  # how many workers for loading data
    max_epoch = 200
    lr = 1e-3  # initial learning rate
    weight_decay = 1e-4

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
