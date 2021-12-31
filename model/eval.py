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
                        help="dimensions of the simulation parameters (default: 1)")
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

    parser.add_argument("--id", type=int, default=0,
                        help="index of the data to evaluate (default: 0)")
    parser.add_argument("--equator", action="store_true", default=False,
                        help="compare the equator patch")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save the npy file")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # train_dataset = MPASDataset(
    #     root=args.root,
    #     train=True,
    #     transform=transforms.Compose([Normalize(), ToTensor()])
    # )
    test_dataset = MPASDataset(
        root=args.root,
        train=False,
        transform=transforms.Compose([Normalize(), ToTensor()])
    )

    kwargs = {"num_workers": 4, "pin_memory": True}
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
    #                           shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, **kwargs)

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
    upAsgnValue = torch.from_numpy(np.load(os.path.join(args.root, "graphM",  "ghtUpAsgnValue1.npy")).astype(np.float32)).to('cuda:0')

    g_model = Generator(graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                        upAsgnIndices, upAsgnValues,
                        args.dsp, args.dspe, args.ch)
    if args.sn:
        g_model = add_sn(g_model)

    if args.gan_loss != "none":
        d_model = Discriminator(graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                                avgPoolAsgnIndices, avgPoolAsgnValues,
                                args.dsp, args.dspe, args.ch)

    l1_criterion = nn.L1Loss()
    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

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

    # evaluating...
    # 1) plot losses
    # fig, ax = plt.subplots()
    # ax.set(xlabel=u"epoch", ylabel=u"loss")
    # plt.plot(train_losses, label="training")
    # # plt.plot(test_losses, label="testing")
    # plt.legend()
    # plt.show()

    g_model.train()     # In BatchNorm, we still want the mean and var calculated from the current instance
    # with torch.no_grad():
    #     sample = test_dataset[args.id]
    #     data = sample["data"].view(1, upAsgnValue.shape[0], 1)
    #     sparams = sample["params"].view(1, args.dsp)
    #     fake_data = g_model(sparams)
    #     fake_data = batch_spmm(upAsgnIdx, upAsgnValue, upAsgnValue.shape[0], graphSizes[0], fake_data)
    #     diff = abs(data.to(g_model.graphBG7_device) - fake_data)
    #     pdb.set_trace()
    #     fake_data = fake_data.view(upAsgnValue.shape[0]).cpu().numpy().astype(np.double) * 12.12 - 0.44
    #     np.save(os.path.join(args.root, "test", "{:04d}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_temperature_fake.npy".format(
    #         args.id + 70, sample["params"][0].item() * 2.5 + 2.5, sample["params"][1].item() * 600.0 + 900.0,
    #         sample["params"][2].item() * .375 + .625, sample["params"][3].item() * 100.0 + 200.0
    #     )), fake_data)
    #     fake_data.tofile(os.path.join(args.root, "test", "{:04d}_{:.5f}_{:.5f}_{:.5f}_{:.5f}_temperature_fake.bin".format(
    #         args.id + 70, sample["params"][0].item() * 2.5 + 2.5, sample["params"][1].item() * 600.0 + 900.0,
    #                  sample["params"][2].item() * .375 + .625, sample["params"][3].item() * 100.0 + 200.0
    #     )))

    equator = np.load(os.path.join(args.root, "../equator_patch.npy"))
    mse = 0.
    psnrs = np.zeros(len(test_loader.dataset))
    max_diff = np.zeros(len(test_loader.dataset))
    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_loader)):
            data = sample["data"]
            sparams = sample["params"]
            data = data.view(upAsgnIdx[0].max() + 1).cpu().numpy().astype(np.double) * 12.12 - 0.44
            # data = data.view(upAsgnIdx[1].max() + 1).cpu().numpy().astype(np.double) * 12.12 - 0.44
            fake_data = g_model(sparams)
            fake_data = batch_spmm(upAsgnIdx, upAsgnValue, upAsgnValue.shape[0], graphSizes[0], fake_data)
            fake_data = fake_data.view(upAsgnIdx[0].max() + 1).cpu().numpy().astype(np.double) * 12.12 - 0.44
            # fake_data = fake_data.view(upAsgnIdx[1].max() + 1).cpu().numpy().astype(np.double) * 12.12 - 0.44

            if args.equator:
                data = data * equator
                data = data[abs(data) > 0]
                fake_data = fake_data * equator
                fake_data = fake_data[abs(fake_data) > 0]
            diff = abs(data - fake_data)
            max_diff[i] = diff.max()
            mse += np.power(data - fake_data, 2.).mean()
            if args.equator:
                psnrs[i] = 20. * np.log10(29.50 - 11.00) - 10. * np.log10(np.power(data - fake_data, 2.).mean())
            else:
                psnrs[i] = 20. * np.log10(1.93 + 30.35) - 10. * np.log10(np.power(data - fake_data, 2.).mean())

            if args.equator:
                print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(29.50 - 11.00) -
                                                10. * np.log10(np.power(data - fake_data, 2.).mean())))
            else:
                print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(1.93 + 30.35) -
                                                10. * np.log10(np.power(data - fake_data, 2.).mean())))

            if args.save:
                np.save(os.path.join(args.root, "test", "{:04d}_temperature_fake.npy".format(
                    i + 70
                )), fake_data)
                fake_data.tofile(os.path.join(args.root, "test", "{:04d}_temperature_fake.bin".format(
                    i + 70
                )))

    if args.equator:
        print("====> PSNR on raw avg {}, std var {}"
          .format(20. * np.log10(29.50 - 11.00) -
                  10. * np.log10(mse / len(test_loader.dataset)), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (29.50 - 11.00), max_diff.std() / (29.50 - 11.00)))
    else:
        # print("====> PSNR on residual {}"
        #       .format(20. * np.log10(12.12 * 2) -
        #               10. * np.log10(mse / len(test_loader.dataset))))
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(1.93 + 30.35) -
                      10. * np.log10(mse / len(test_loader.dataset)),  psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (1.93 + 30.35), max_diff.std() / (1.93 + 30.35)))


if __name__ == "__main__":
  main(parse_args())
