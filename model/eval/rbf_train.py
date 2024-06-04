import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from rbf_utils import *

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

    parser.add_argument("--lr", type=float, default=1e-3,
                            help="learning rate (default: 1e-3)")
    parser.add_argument("--beta1", type=float, default=0.0,
                            help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--epochs", type=int, default=10000,
                            help="number of epochs to train")

    parser.add_argument("--log-every", type=int, default=10,
                            help="log training status every given given number of epochs (default: 10)")
    parser.add_argument("--check-every", type=int, default=2000,
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

    params = np.load(os.path.join(args.root, "train", "params.npy"))[:, 1:]
    # params min [0.0, 300.0, 0.25, 100.0]
    #        max [5.0, 1500.0, 1.0, 300.0]
    params = (params.astype(np.float32) - np.array([2.5, 900.0, .625, 200.0], dtype=np.float32)) / \
             np.array([2.5, 600.0, .375, 100.0], dtype=np.float32)
    params = torch.from_numpy(params).cuda()


    fh = open(os.path.join(args.root, "train", "npyNames.txt"))
    filenames = []
    for line in fh:
        filenames.append(line)

    data = []
    N = len(filenames)
    for idx in range(N):
        filename = filenames[idx]
        filename = filename.strip("\r\n")
        data.append(np.load(os.path.join(args.root, "train", filename)))
    data = np.asarray(data).reshape((N, -1))
    dim = data.shape[1]

    dmin = -1.93
    dmax = 30.35
    data = (data.astype(np.float32) - (dmin + dmax) / 2.0) / ((dmax - dmin) / 2.0)

    #######################################################################
    # Perform the **Kernel interpolation**

    K_xx = gaussian_kernel(params, params)

    # Initialize the loss function
    mse_criterion = nn.MSELoss()

    for i in range(dim // args.batch_size):
        sub_data = torch.from_numpy(data[:, args.batch_size * i:args.batch_size * (i+1)]).cuda()
        a = torch.zeros(N, args.batch_size, device="cuda", requires_grad=True)
        # a = Variable(torch.from_numpy(np.load(
        #     os.path.join(args.root, "rbf_weights", "batch_{:d}.npy".format(i)))).cuda(), requires_grad=True)
        optimizer = optim.Adagrad([a], lr=args.lr)
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            cost = K_xx @ a - sub_data
            loss = mse_criterion(cost, torch.zeros_like(cost))
            loss.backward()
            optimizer.step()

            # log training status
            if epoch % args.log_every == 0:
                print("Train Epoch: {} MSE_Loss: {:.6f}".format(
                    epoch, loss.detach().item()))
            if (epoch + 1) % args.check_every == 0:
                print("=> saving checkpoint at epoch {}".format(epoch + 1))
                np.save(os.path.join(args.root, "rbf_weights", "epoch_{:d}_batch{:d}.npy".format(epoch + 1, i)), a.detach().cpu().numpy())

if __name__ == "__main__":
    main(parse_args())
