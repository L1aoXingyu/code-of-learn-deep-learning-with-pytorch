__author__ = 'SherlockLiao'

import torch
from torch.utils.data import Dataset
import h5py


class h5Dataset(Dataset):

    def __init__(self, h5py_list):
        label_file = h5py.File(h5py_list[0], 'r')
        self.label = torch.from_numpy(label_file['label'].value)
        self.nSamples = self.label.size(0)
        temp_dataset = torch.FloatTensor()
        for file in h5py_list:
            h5_file = h5py.File(file, 'r')
            dataset = torch.from_numpy(h5_file['data'].value)
            temp_dataset = torch.cat((temp_dataset, dataset), 1)

        self.dataset = temp_dataset

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        data = self.dataset[index]
        label = self.label[index]
        return (data, label)
