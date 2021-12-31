import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from layer import *
from utils import *

import pdb

class BasicBlockGenerator(nn.Module):
    def __init__(self, in_features, out_features,
                 adjValue, edgeOne, E_start, E_end,
                 upAsgnIdx, upAsgnValue, nextDim, device,
                 activation=F.relu, upsample=True):
        super(BasicBlockGenerator, self).__init__()

        self.adjValue = adjValue
        self.edgeOne = edgeOne
        self.E_start = E_start
        self.E_end = E_end

        self.upAsgnIdx = upAsgnIdx.to(device)
        self.upAsgnValue = upAsgnValue.to(device)
        self.nextDim = nextDim
        self.activation = activation
        self.upsample = upsample
        self.conv_res = None
        if self.upsample or in_features != out_features:
            self.conv_res = nn.Conv1d(in_features, out_features, 1, 1, 0, bias=True)

        self.bn0 = nn.BatchNorm1d(in_features)
        self.bn1 = nn.BatchNorm1d(out_features)

        self.conv0 = ECC(in_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)
        self.conv1 = ECC(out_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)

    def forward(self, x):
        residual = x
        if self.upsample:
            residual = batch_spmm(self.upAsgnIdx, self.upAsgnValue, self.nextDim, residual.shape[1], residual)
        if self.conv_res:
            residual = self.conv_res(residual.transpose(1, 2)).transpose(1, 2)

        out = self.bn0(x.transpose(1, 2)).transpose(1, 2)
        del x
        out = self.activation(out, inplace=True)
        if self.upsample:
            out = batch_spmm(self.upAsgnIdx, self.upAsgnValue, self.nextDim, out.shape[1], out)
        out = self.conv0(out)

        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out, inplace=True)
        out = self.conv1(out)

        return out + residual

class LastBlockGenerator(nn.Module):
    def __init__(self, in_features, out_features,
                 adjValue, edgeOne, E_start, E_end,
                 device, activation=F.relu):
        super(LastBlockGenerator, self).__init__()

        self.adjValue = adjValue
        self.edgeOne = edgeOne
        self.E_start = E_start
        self.E_end = E_end

        self.activation = activation
        self.conv = ECC(in_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)
        self.bn0 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        out = self.bn0(x.transpose(1, 2)).transpose(1, 2)
        del x
        out = self.activation(out, inplace=True)
        out = self.conv(out)
        return out

class BasicBlockDiscriminator(nn.Module):
    def __init__(self, in_features, out_features,
                 adjValue, edgeOne, E_start, E_end,
                 avgPoolAsgnIdx, avgPoolAsgnValue, nextDim, device,
                 activation=F.leaky_relu, downsample=True):
        super(BasicBlockDiscriminator, self).__init__()

        self.adjValue = adjValue
        self.edgeOne = edgeOne
        self.E_start = E_start
        self.E_end = E_end

        if downsample:
            self.avgPoolAsgnIdx = avgPoolAsgnIdx.to(device)
            self.avgPoolAsgnValue = avgPoolAsgnValue.to(device)
        self.nextDim = nextDim
        self.activation = activation
        self.downsample = downsample
        self.conv_res = None
        if self.downsample or in_features != out_features:
            self.conv_res = nn.Conv1d(in_features, out_features, 1, 1, 0, bias=True)

        self.conv0 = ECC(in_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)
        self.conv1 = ECC(out_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)

    def forward(self, x):
        residual = x
        if self.conv_res is not None:
            residual = self.conv_res(residual.transpose(1, 2)).transpose(1, 2)
        if self.downsample:
            residual = batch_spmm(self.avgPoolAsgnIdx, self.avgPoolAsgnValue, self.nextDim, residual.shape[1], residual)

        out = self.activation(x, 0.2)
        del x
        out = self.conv0(out)

        out = self.activation(out, 0.2, inplace=True)
        out = self.conv1(out)

        if self.downsample:
            out = batch_spmm(self.avgPoolAsgnIdx, self.avgPoolAsgnValue, self.nextDim, out.shape[1], out)

        return out + residual

class FirstBlockDiscriminator(nn.Module):
    def __init__(self, in_features, out_features,
                 adjValue, edgeOne, E_start, E_end,
                 avgPoolAsgnIdx, avgPoolAsgnValue, nextDim, device,
                 activation=F.leaky_relu):
        super(FirstBlockDiscriminator, self).__init__()

        self.adjValue = adjValue
        self.edgeOne = edgeOne
        self.E_start = E_start
        self.E_end = E_end

        self.avgPoolAsgnIdx = avgPoolAsgnIdx.to(device)
        self.avgPoolAsgnValue = avgPoolAsgnValue.to(device)
        self.nextDim = nextDim
        self.activation = activation
        self.conv_res = nn.Conv1d(in_features, out_features, 1, 1, 0, bias=True)

        self.conv0 = ECC(in_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)
        self.conv1 = ECC(out_features, out_features, self.adjValue, self.edgeOne, self.E_start, self.E_end)


    def forward(self, x):
        residual = self.conv_res(x.transpose(1, 2)).transpose(1, 2)
        residual = batch_spmm(self.avgPoolAsgnIdx, self.avgPoolAsgnValue, self.nextDim, residual.shape[1], residual)

        out = self.conv0(x)
        del x
        out = self.activation(out, 0.2, inplace=True)
        out = self.conv1(out)

        out = batch_spmm(self.avgPoolAsgnIdx, self.avgPoolAsgnValue, self.nextDim, out.shape[1], out)

        return out + residual


