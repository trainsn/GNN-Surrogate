# main file

from __future__ import absolute_import, division, print_function

import os
import argparse
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("/root/apex")
from apex import amp

from mpas import *
from generator import Generator
from discriminator import Discriminator
from resblock import *
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
    parser.add_argument("--num-run", type=int, default=0,
                        help="the number of Ensemble Runs")
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

    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate (default: 5e-5)")
    parser.add_argument("--d-lr", type=float, default=2e-4,
                        help="learning rate of the discriminator (default: 2e-4)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--opt-level", default='O2',
                        help='amp opt_level, default="O2"')
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of epochs to train")

    parser.add_argument("--log-every", type=int, default=10,
                        help="log training status every given given number of epochs (default: 10)")
    parser.add_argument("--check-every", type=int, default=200,
                        help="save checkpoint every given number of epochs ")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train_dataset = MPASDataset(
        root=args.root,
        train=True,
        num_run=args.num_run,
        transform=transforms.Compose([Normalize(), ToTensor()])
    )
    # test_dataset = MPASDataset(
    #     root=args.root,
    #     train=False,
    #     data_len=1000,
    #     transform=transforms.Compose([Normalize(), ToTensor()])
    # )

    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **kwargs)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
    #                          shuffle=False, **kwargs)

    # model
    def weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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

    g_model = Generator(graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                        upAsgnIndices, upAsgnValues, args.dsp, args.dspe, args.ch)
    g_model.apply(weights_init)
    if args.sn:
        g_model = add_sn(g_model)

    if args.gan_loss != "none":
        d_model = Discriminator(graphSizes, adjValues, edgeOnes, E_starts, E_ends,
                                avgPoolAsgnIndices, avgPoolAsgnValues, args.dsp, args.dspe, args.ch)
        d_model.apply(weights_init)
        if args.sn:
            d_model = add_sn(d_model)

    l1_criterion = nn.L1Loss()
    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2))
    if args.gan_loss != "none":
        d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
                                 betas=(args.beta1, args.beta2))

    if args.gan_loss == "none":
        g_model, g_optimizer = amp.initialize(g_model, g_optimizer, opt_level=args.opt_level)
    else:
        [g_model, d_model], [g_optimizer, d_optimizer] = amp.initialize(
            [g_model, d_model], [g_optimizer, d_optimizer], opt_level=args.opt_level, num_losses=2
        )

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        g_model.load_state_dict(checkpoint["g_model_state_dict"])
        # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        if args.gan_loss != "none":
            d_model.load_state_dict(checkpoint["d_model_state_dict"])
            # d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
            d_losses = checkpoint["d_losses"]
            g_losses = checkpoint["g_losses"]
        train_losses = checkpoint["train_losses"]
        test_losses = checkpoint["test_losses"]
        print("=> loaded checkpoint {} (epoch {})"
              .format(args.resume, checkpoint["epoch"]))
        del checkpoint

    # main loop
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # training...
        g_model.train()
        if args.gan_loss != "none":
            d_model.train()
        train_loss = 0.
        for i, sample in enumerate(train_loader):
            data = sample["data"].to(g_model.graphBG7_device)

            sparams = sample["params"]
            g_optimizer.zero_grad()
            fake_data = g_model(sparams)
            # torch.cuda.empty_cache()

            loss = 0.

            # l1 loss
            l1_loss = l1_criterion(data, fake_data)
            if args.gan_loss == "none":
                del data
                del fake_data
            loss += l1_loss

            # loss of generator
            if args.gan_loss != "none":
                g_optimizer.zero_grad()
                fake_decision = d_model(sparams, fake_data)

                if args.gan_loss == "vanilla":
                    g_loss = args.gan_loss_weight * torch.mean(F.softplus(-fake_decision))
                elif args.gan_loss == "wgan":
                    g_loss = args.gan_loss_weight * torch.mean(-fake_decision)
                elif args.gan_loss == "lsgan":
                    g_loss = args.gan_loss_weight * torch.mean((fake_decision - 1) ** 2)
                del fake_decision
                loss += g_loss.to(g_model.graphBG7_device)
            with amp.scale_loss(loss, g_optimizer, loss_id=0) as loss_scaled:
                loss_scaled.backward()

            g_optimizer.step()
            train_loss += loss.detach().item() * len(sparams)

            # gan loss
            if args.gan_loss != "none":
                # update discriminator
                d_optimizer.zero_grad()
                decision = d_model(sparams, data)
                del data
                if args.gan_loss == "vanilla":
                    d_loss_real = torch.mean(F.softplus(-decision))
                elif args.gan_loss == "wgan":
                    d_loss_real = torch.mean(-decision)
                elif args.gan_loss == "lsgan":
                    d_loss_real = torch.mean((decision - 1) ** 2)
                del decision

                fake_decision = d_model(sparams, fake_data.detach())
                del fake_data
                if args.gan_loss == "vanilla":
                    d_loss_fake = torch.mean(F.softplus(fake_decision))
                elif args.gan_loss == "wgan":
                    d_loss_fake = torch.mean(fake_decision)
                elif args.gan_loss == "lsgan":
                    d_loss_fake = torch.mean(fake_decision ** 2)
                del fake_decision

                # with amp.scale_loss(d_loss_real, d_optimizer, loss_id=1) as d_loss_real_scaled:
                #     d_loss_real_scaled.backward()
                # with amp.scale_loss(d_loss_fake, d_optimizer, loss_id=2) as d_loss_fake_scaled:
                #     d_loss_fake_scaled.backward()
                d_loss = d_loss_real + d_loss_fake
                with amp.scale_loss(d_loss, d_optimizer, loss_id=1) as d_loss_scaled:
                    d_loss_scaled.backward()
                d_optimizer.step()

            # log training status
            if i % args.log_every == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tL1_Loss: {:.6f}".format(
                    epoch, i * len(sparams), len(train_loader.dataset),
                           100. * i / len(train_loader),
                    l1_loss.detach().item()))
                if args.gan_loss != "none":
                    print("DLoss: {:.6f}, GLoss: {:.6f}".format(
                        d_loss.detach().item(), g_loss.detach().item()))
                    d_losses.append(d_loss.detach().item())
                    g_losses.append(g_loss.detach().item())
                train_losses.append(l1_loss.detach().item())

            del l1_loss
            del loss
            if args.gan_loss != "none":
                del g_loss
                del d_loss
                del d_loss_real
                del d_loss_fake
            # torch.cuda.empty_cache()

        print("====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)))

        # testing...
        # g_model.eval()
        # if args.gan_loss != "none":
        #     d_model.eval()
        # test_loss = 0.
        # with torch.no_grad():
        #     for i, sample in enumerate(test_loader):
        #         data = sample["data"].to('cuda:0')
        #         sparams = sample["sparams"].to('cuda:0')
        #         fake_data = g_model(sparams)
        #         test_loss += l1_criterion(data, fake_data).item()
        #
        #         if i == 0:
        #             n = min(len(sparams), 5)
        #
        # test_losses.append(test_loss / len(test_loader.dataset))
        # print("====> Epoch: {} Test set loss: {:.4f}".format(
        #     epoch, test_losses[-1]))

        # saving...
        if (epoch + 1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            if args.gan_loss != "none":
                torch.save({"epoch": epoch + 1,
                            "g_model_state_dict": g_model.state_dict(),
                            # "g_optimizer_state_dict": g_optimizer.state_dict(),
                            "d_model_state_dict": d_model.state_dict(),
                            # "d_optimizer_state_dict": d_optimizer.state_dict(),
                            "d_losses": d_losses,
                            "g_losses": g_losses,
                            "train_losses": train_losses,
                            "test_losses": test_losses},
                           os.path.join(args.root, "model_" + str(epoch + 1) + ".pth.tar"))
            else:
                torch.save({"epoch": epoch + 1,
                            "g_model_state_dict": g_model.state_dict(),
                            # "g_optimizer_state_dict": g_optimizer.state_dict(),
                            "train_losses": train_losses,
                            "test_losses": test_losses},
                           os.path.join(args.root, "model_" + str(epoch + 1) + ".pth.tar"))

            torch.save(g_model.state_dict(),
                       os.path.join(args.root, "model_" + str(epoch + 1) + ".pth"))

if __name__ == "__main__":
  main(parse_args())
