# Graph convolution

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils import *
from torch_sparse import spspmm

import pdb

class ECC(nn.Module):
    def __init__(self, in_features, out_features,
                 adjValue, edgeOne, E_start, E_end):
        super(ECC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.adjValue = adjValue
        self.edgeOne = edgeOne
        self.E_start = E_start
        self.E_end = E_end
        self.edgeNumAttr = 7
        self.E = [self.edgeOne[i].shape[0] for i in range(self.edgeNumAttr)]
        self.N = max([self.E_start[i][1].max() + 1 for i in range(self.edgeNumAttr)])

        self.UHL = nn.Linear(in_features, out_features)
        self.ULL = nn.Linear(in_features, out_features)
        self.UW = nn.Linear(in_features, out_features)
        self.UE = nn.Linear(in_features, out_features)
        self.UU = nn.Linear(in_features, out_features)
        self.UD = nn.Linear(in_features, out_features)
        self.US = nn.Linear(in_features, out_features)
        self.kernels = [self.UHL, self.ULL, self.UW, self.UE, self.UU, self.UD, self.US]

    def forward(self, input):
        for i in range(self.edgeNumAttr):
            x = self.kernels[i](input)  # N x C_out
            x2 = batch_spmm(self.E_end[i], self.adjValue[i], self.E[i], self.N, x)  # E x C_out
            del x
            if i == 0:
                out = batch_spmm(torch.vstack((self.E_start[i][1], self.E_start[i][0])), self.edgeOne[i], self.N, self.E[i], x2)    # N x C_out
            else:
                out += batch_spmm(torch.vstack((self.E_start[i][1], self.E_start[i][0])), self.edgeOne[i], self.N, self.E[i], x2)    # N x C_out
        return out
