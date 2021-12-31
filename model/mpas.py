# MPAS dataset

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MPASDataset(Dataset):
    def __init__(self, root, train=True, num_run=0, data_len=0, transform=None):
        if train:
            if num_run > 0:
                fh = open(os.path.join(root, "train", "npyNames_" + str(num_run) + ".txt"))
            else:
                fh = open(os.path.join(root, "train", "npyNames.txt"))
        else:
            fh = open(os.path.join(root, "../residual/test", "npyNames.txt"))
        filenames = []
        for line in fh:
            filenames.append(line)

        self.root = root
        self.train = train
        self.num_run = num_run
        self.data_len = data_len
        self.transform = transform
        self.filenames = filenames
        if self.train:
            if num_run > 0:
                self.params = np.load(os.path.join(root, "train/params_" + str(num_run) + ".npy"))
            else:
                self.params = np.load(os.path.join(root, "train/params.npy"))
        else:
            self.params = np.load(os.path.join(root, "../residual/test/params.npy"))

    def __len__(self):
        if self.data_len:
            return self.data_len
        else:
            return len(self.filenames)

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()

        filename = self.filenames[index]
        filename = filename.strip("\r\n")
        if self.train:
            data_name = os.path.join(self.root, "train", filename)
        else:
            data_name = os.path.join(self.root, "../residual/test", filename)
        data = np.load(data_name)

        params = self.params[index, 1:]
        sample = {"data": data, "params": params}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        data = sample["data"]
        params = sample["params"]

        # data min -12.557852
        #      max 11.666099
        data = (data.astype(np.float32) + 0.44) / 12.12

        # params min [0.0, 300.0, 0.25, 100.0]
        #        max [5.0, 1500.0, 1.0, 300.0]
        params = (params.astype(np.float32) - np.array([2.5, 900.0, .625, 200.0], dtype=np.float32)) / \
                 np.array([2.5, 600.0, .375, 100.0], dtype=np.float32)

        return {"data": data, "params": params}

class ToTensor(object):
    def __call__(self, sample):
        data = sample["data"]
        params = sample["params"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [N, 1]
        data = data[:, None]
        assert data.shape[1] == 1
        return {"data": torch.from_numpy(data),
                "params": torch.from_numpy(params)}
