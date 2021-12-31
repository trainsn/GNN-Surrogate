# Discriminator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import FirstBlockDiscriminator, BasicBlockDiscriminator

import pdb

class Discriminator(nn.Module):
    def __init__(self, graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                 avgPoolAsgnIndices, avgPoolAsgnValues,
                 dsp=4, dspe=512, ch=64):
        # dsp  - dimensions of the simulation parameters
        # dspe - dimensions of the simulation parameters' encode
        # ch   - channel multiplier
        super(Discriminator, self).__init__()

        self.dsp, self.dspe = dsp, dspe
        self.ch = ch
        self.cgs = graphSizes[-1]

        # parameters_subnet
        # self.params_subnet_device = torch.device("cuda:0")
        # self.param_subnet = nn.Sequential(
        #     nn.Linear(dsp, dspe), nn.ReLU(),
        #     nn.Linear(dspe, dspe), nn.ReLU(),
        #     nn.Linear(dspe, dspe), nn.ReLU(),
        #     nn.Linear(dspe, ch * 384), nn.ReLU()
        # ).to(self.params_subnet_device)

        # graph block discriminators
        self.graphBD1_device = torch.device("cuda:0")
        self.graphBD2_device = torch.device("cuda:0")
        self.graphBD3_device = torch.device("cuda:0")
        self.graphBD4_device = torch.device("cuda:0")
        self.graphBD5_device = torch.device("cuda:0")
        self.graphBD6_device = torch.device("cuda:0")
        self.graphBD7_device = torch.device("cuda:0")
        self.graphBD8_device = torch.device("cuda:0")

        self.graphBD1 = FirstBlockDiscriminator(1, ch,
                                    adjValues[0], edgeOnes[0], E_starts[0], E_ends[0],
                                    avgPoolAsgnIndices[0], avgPoolAsgnValues[0], graphSizes[1],
                                    self.graphBD1_device).to(self.graphBD1_device)
        self.graphBD2 = BasicBlockDiscriminator(ch, ch,
                                    adjValues[1], edgeOnes[1], E_starts[1], E_ends[1],
                                    avgPoolAsgnIndices[1], avgPoolAsgnValues[1], graphSizes[2],
                                    self.graphBD2_device).to(self.graphBD2_device)  # 1->3
        self.graphBD3 = BasicBlockDiscriminator(ch, ch * 2,
                                    adjValues[2], edgeOnes[2], E_starts[2], E_ends[2],
                                    avgPoolAsgnIndices[2], avgPoolAsgnValues[2], graphSizes[3],
                                    self.graphBD3_device).to(self.graphBD3_device)  # 3->5
        self.graphBD4 = BasicBlockDiscriminator(ch * 2, ch * 4,
                                    adjValues[3], edgeOnes[3], E_starts[3], E_ends[3],
                                    avgPoolAsgnIndices[3], avgPoolAsgnValues[3], graphSizes[4],
                                    self.graphBD4_device).to(self.graphBD4_device)  # 5->6
        self.graphBD5 = BasicBlockDiscriminator(ch * 4, ch * 8,
                                    adjValues[4], edgeOnes[4], E_starts[4], E_ends[4],
                                    avgPoolAsgnIndices[4], avgPoolAsgnValues[4], graphSizes[5],
                                    self.graphBD5_device).to(self.graphBD5_device)  # 6->8
        self.graphBD6 = BasicBlockDiscriminator(ch * 8, ch * 8,
                                    adjValues[5], edgeOnes[5], E_starts[5], E_ends[5],
                                    avgPoolAsgnIndices[5], avgPoolAsgnValues[5], graphSizes[6],
                                    self.graphBD6_device).to(self.graphBD6_device)  # 8->10
        self.graphBD7 = BasicBlockDiscriminator(ch * 8, ch * 16,
                                    adjValues[6], edgeOnes[6], E_starts[6], E_ends[6],
                                    avgPoolAsgnIndices[6], avgPoolAsgnValues[6], graphSizes[7],
                                    self.graphBD7_device).to(self.graphBD7_device)
        self.graphBD8 = BasicBlockDiscriminator(ch * 16, ch * 16,
                                    adjValues[7], edgeOnes[7], E_starts[7], E_ends[7],
                                    None, None, None,
                                    self.graphBD8_device, downsample=False).to(self.graphBD8_device)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # output subnets
        self.out_subnet_device = torch.device("cuda:0")
        self.out_subnet = nn.Sequential(
            nn.Linear(ch * 16 * graphSizes[-1], 1)
        ).to(self.out_subnet_device)

    def forward(self, sp, x):
        # sp = self.param_subnet(sp.to(self.params_subnet_device))

        # x = self.graphBD0(x.to(self.graphBD0_device))
        x = self.graphBD1(x.to(self.graphBD1_device))
        x = self.graphBD2(x.to(self.graphBD2_device))
        x = self.graphBD3(x.to(self.graphBD3_device))
        x = self.graphBD4(x.to(self.graphBD4_device))
        x = self.graphBD5(x.to(self.graphBD5_device))
        x = self.graphBD6(x.to(self.graphBD6_device))
        x = self.graphBD7(x.to(self.graphBD7_device))
        x = self.graphBD8(x.to(self.graphBD8_device))
        x = self.activation(x)
        x = x.view(-1, self.cgs * self.ch * 16)
        # x = torch.sum(x, 1)    # global sum pooling

        out = self.out_subnet(x.to(self.out_subnet_device))    # shape = [batch_size, 1]
        # out += torch.sum(sp.to(self.out_subnet_device) * x, 1, keepdim=True)

        return out