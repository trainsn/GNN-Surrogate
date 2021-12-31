import os
import argparse
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from mpas import *
from generator import Generator
from discriminator import Discriminator
from graph import load_graph
from utils import *

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--dsp", type=int, default=4,
                        help="dimensions of the simulation parameters (default: 4)")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--ch", type=int, default=64,
                        help="channel multiplier (default: 64)")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--gan-loss", type=str, default="none",
                        help="gan loss (default: none): none, vanilla, wgan")
    parser.add_argument("--gan-loss-weight", type=float, default=1e-2,
                        help="weight of the gan loss (default: 1e-2)")

    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")

    parser.add_argument("--bwsa", type=float, default=2.5,
                        help="config_bulk_wind_stress_amp (BwsA) (default 2.5)")
    parser.add_argument("--kappa", type=float, default=900.0,
                        help="config_gm_constant_kappa (default 900.0)")
    parser.add_argument("--cvmix", type=float, default=0.625,
                        help="config_cvmix_kpp_criticalbulkrichardsonnumber (default 0.625)")
    parser.add_argument("--mom", type=float, default=200.0,
                        help="config_mom_del2 (default: 200.0)")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    graphSizes, adjValues, edgeOnes, E_starts, E_ends, \
    avgPoolAsgnIndices, avgPoolAsgnValues, upAsgnIndices, upAsgnValues = \
        load_graph(os.path.join(args.root, "graphM"))
    upAsgnIdx = torch.from_numpy(np.load(os.path.join(args.root, "graphM", "ghtUpAsgnIdx1.npy"))).type(torch.LongTensor).to('cuda:0')
    upAsgnValue = torch.from_numpy(np.load(os.path.join(args.root, "graphM", "ghtUpAsgnValue1.npy")).astype(np.float32)).to('cuda:0')

    g_model = Generator(graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                        upAsgnIndices, upAsgnValues,
                        args.dsp, args.dspe, args.ch)
    if args.sn:
        g_model = add_sn(g_model)

    if args.gan_loss != "none":
        d_model = Discriminator(graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                                avgPoolAsgnIndices, avgPoolAsgnValues,
                                args.dsp, args.dspe, args.ch)
    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device("cuda:0"))
        args.start_epoch = checkpoint["epoch"]
        g_model.load_state_dict(checkpoint["g_model_state_dict"])
        # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        if args.gan_loss != "none":
            # d_model.load_state_dict(checkpoint["d_model_state_dict"])
            # d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
            d_losses = checkpoint["d_losses"]
            g_losses = checkpoint["g_losses"]
        train_losses = checkpoint["train_losses"]
        test_losses = checkpoint["test_losses"]
        print("=> loaded checkpoint {} (epoch {})"
                .format(args.resume, checkpoint["epoch"]))

    equator = np.load(os.path.join(args.root, "../equator_patch.npy"))
    BwsAMin, BwsAMax = 0.0, 5.0
    kappaMin, kappaMax = 300.0, 1500.0
    cvmixMin, cvmixMax = 0.25, 1.0
    momMin, momMax = 100.0, 300.0

    g_model.train()  # In BatchNorm, we still want the mean and var calculated from the current instance
    with torch.no_grad():
        sparams = torch.zeros(1, 4)
        sparams[0, 0] = (args.bwsa - BwsAMin) / (BwsAMax - BwsAMin)
        sparams[0, 1] = (args.kappa - kappaMin) / (kappaMax - kappaMin)
        sparams[0, 2] = (args.cvmix - cvmixMin) / (cvmixMax - cvmixMin)
        sparams[0, 3] = (args.mom - momMin) / (momMax - momMin)

        fake_data = g_model(sparams)
        fake_data = batch_spmm(upAsgnIdx, upAsgnValue, upAsgnValue.shape[0], graphSizes[0], fake_data)
        fake_data = fake_data.view(upAsgnIdx[0].max() + 1).cpu().numpy().astype(np.double) * 12.12 - 0.44

        reference = np.load(os.path.join(args.root, "../train", "0059_3.13326_1195.80632_0.59768_192.34201_temperature.npy"))
        fake_data = fake_data + reference

        np.save(os.path.join(args.root, "../case", "BwsA{:.2f}_temperature_fake.npy".format(
            args.bwsa
        )), fake_data)
        fake_data.tofile(os.path.join(args.root, "../case", "BwsA{:.2f}_temperature_fake.bin".format(
            args.bwsa
        )))


if __name__ == "__main__":
  main(parse_args())
