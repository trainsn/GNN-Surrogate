# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockGenerator, LastBlockGenerator
from layer import *

import pdb

class Generator(nn.Module):
    def __init__(self, graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                 upAsgnIndices, upAsgnValues,
                 dsp=4, dspe=512, ch=64):
        # dsp - dimensions of the simulation parameters
        # dspe - dimensions of the simulation parameters' encode
        # ch - channel multiplier
        super(Generator, self).__init__()
        self.dsp, self.dspe = dsp, dspe
        self.ch = ch
        self.cgs = graphSizes[-1]

        # parameters subnet
        self.params_subnet_device = torch.device("cuda:0")
        self.params_subnet = nn.Sequential(
            nn.Linear(dsp, dspe // 2), nn.ReLU(),
            nn.Linear(dspe // 2, dspe), nn.ReLU(),
            nn.Linear(dspe, dspe), nn.ReLU(),
            nn.Linear(dspe, ch * 16 * graphSizes[-1], bias=True)
        ).to(self.params_subnet_device)

        # graph block generators
        self.graphBG0_device = torch.device("cuda:0")
        self.graphBG1_device = torch.device("cuda:0")
        self.graphBG2_device = torch.device("cuda:0")
        self.graphBG3_device = torch.device("cuda:0")
        self.graphBG4_device = torch.device("cuda:0")
        self.graphBG5_device = torch.device("cuda:0")
        self.graphBG6_device = torch.device("cuda:0")
        self.graphBG7_device = torch.device("cuda:0")
        # self.graphBG8_device = torch.device("cuda:0")
        self.graphBG0 = BasicBlockGenerator(ch * 16, ch * 16,
                                            adjValues[-2], edgeOnes[-2], E_starts[-2], E_ends[-2],
                                            upAsgnIndices[-1], upAsgnValues[-1], graphSizes[-2],
                                            self.graphBG0_device).to(self.graphBG0_device)  # 12->11
        self.graphBG1 = BasicBlockGenerator(ch * 16, ch * 8,
                                            adjValues[-3], edgeOnes[-3], E_starts[-3], E_ends[-3],
                                            upAsgnIndices[-2], upAsgnValues[-2], graphSizes[-3],
                                            self.graphBG1_device).to(self.graphBG1_device)  # 11->10
        self.graphBG2 = BasicBlockGenerator(ch * 8, ch * 8,
                                            adjValues[-4], edgeOnes[-4], E_starts[-4], E_ends[-4],
                                            upAsgnIndices[-3], upAsgnValues[-3], graphSizes[-4],
                                            self.graphBG2_device).to(self.graphBG2_device)  # 10->8
        self.graphBG3 = BasicBlockGenerator(ch * 8, ch * 4,
                                            adjValues[-5], edgeOnes[-5], E_starts[-5], E_ends[-5],
                                            upAsgnIndices[-4], upAsgnValues[-4], graphSizes[-5],
                                            self.graphBG3_device).to(self.graphBG3_device)  # 8->6
        self.graphBG4 = BasicBlockGenerator(ch * 4, ch * 2,
                                            adjValues[-6], edgeOnes[-6], E_starts[-6], E_ends[-6],
                                            upAsgnIndices[-5], upAsgnValues[-5], graphSizes[-6],
                                            self.graphBG4_device).to(self.graphBG4_device)  # 6->5
        self.graphBG5 = BasicBlockGenerator(ch * 2, ch,
                                            adjValues[-7], edgeOnes[-7], E_starts[-7], E_ends[-7],
                                            upAsgnIndices[-6], upAsgnValues[-6], graphSizes[-7],
                                            self.graphBG5_device).to(self.graphBG5_device)  # 5->3
        self.graphBG6 = BasicBlockGenerator(ch, ch,
                                            adjValues[-8], edgeOnes[-8], E_starts[-8], E_ends[-8],
                                            upAsgnIndices[-7], upAsgnValues[-7], graphSizes[-8],
                                            self.graphBG6_device).to(self.graphBG6_device)  # 3->1
        self.graphBG7 = LastBlockGenerator(ch, 1, adjValues[-8], edgeOnes[-8], E_starts[-8], E_ends[-8],
                                           self.graphBG7_device).to(self.graphBG7_device)

        self.tanh = nn.Tanh()

    def forward(self, sp):
        sp = self.params_subnet(sp.to(self.params_subnet_device))
        x = sp.view(sp.size(0), self.cgs, self.ch * 16)
        del sp
        x = self.graphBG0(x.to(self.graphBG0_device))
        x = self.graphBG1(x.to(self.graphBG1_device))
        x = self.graphBG2(x.to(self.graphBG2_device))
        x = self.graphBG3(x.to(self.graphBG3_device))
        x = self.graphBG4(x.to(self.graphBG4_device))
        x = self.graphBG5(x.to(self.graphBG5_device))
        x = self.graphBG6(x.to(self.graphBG6_device))
        x = self.graphBG7(x.to(self.graphBG7_device))
        # x = self.graphBG8(x.to(self.graphBG8_device))
        x = self.tanh(x)

        return x