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

    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--epoch", type=int, default=10000,
                        help="model epoch")

    parser.add_argument("--equator", action="store_true", default=False,
                        help="compare the equator patch")

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

    params_test = np.load(os.path.join(args.root, "test", "params.npy"))[:, 1:]
    # params min [0.0, 300.0, 0.25, 100.0]
    #        max [5.0, 1500.0, 1.0, 300.0]
    params_test = (params_test.astype(np.float32) - np.array([2.5, 900.0, .625, 200.0], dtype=np.float32)) / \
             np.array([2.5, 600.0, .375, 100.0], dtype=np.float32)
    params_test = torch.from_numpy(params_test).cuda()

    params_train = np.load(os.path.join(args.root, "train", "params.npy"))[:, 1:]
    # params min [0.0, 300.0, 0.25, 100.0]
    #        max [5.0, 1500.0, 1.0, 300.0]
    params_train = (params_train.astype(np.float32) - np.array([2.5, 900.0, .625, 200.0], dtype=np.float32)) / \
                   np.array([2.5, 600.0, .375, 100.0], dtype=np.float32)
    params_train = torch.from_numpy(params_train).cuda()

    fh = open(os.path.join(args.root, "test", "npyNames.txt"))
    filenames = []
    for line in fh:
        filenames.append(line)

    data = []
    N = len(filenames)
    for idx in range(N):
        filename = filenames[idx]
        filename = filename.strip("\r\n")
        data.append(np.load(os.path.join(args.root, "test", filename)))
    data = np.asarray(data).reshape((N, -1)).astype(np.float32)
    dim = data.shape[1]

    dmin = -1.93
    dmax = 30.35

    K_tx = gaussian_kernel(params_test, params_train)

    inter = []
    with torch.no_grad():
        for i in range(dim // args.batch_size):
            a = torch.from_numpy(np.load(
                os.path.join(args.root, "rbf_weights", "epoch_{:d}_batch{:d}.npy".format(args.epoch, i)))).cuda()
            mean_t = K_tx @ a
            inter.append(mean_t.cpu().numpy())
    inter = np.hstack(inter)
    inter = inter * ((dmax - dmin) / 2.0) + (dmin + dmax) / 2.0

    equator = np.load(os.path.join(args.root, "equator_patch.npy"))

    mse = 0.
    psnrs = np.zeros(inter.shape[0])
    max_diff = np.zeros(inter.shape[0])
    for i in range(inter.shape[0]):
        sub_data = data[i]
        sub_inter = inter[i]

        if args.equator:
            sub_data = sub_data * equator
            sub_data = sub_data[sub_data > 0]
            sub_inter = sub_inter * equator
            sub_inter = sub_inter[sub_inter > 0]

        diff = abs(sub_data - sub_inter)
        max_diff[i] = diff.max()
        mse += np.power(sub_data - sub_inter, 2.).mean()

        if args.equator:
            psnrs[i] = 20. * np.log10(29.50 - 11.00) - 10. * np.log10(np.power(sub_data - sub_inter, 2.).mean())
        else:
            psnrs[i] = 20. * np.log10(1.93 + 30.35) - 10. * np.log10(np.power(sub_data - sub_inter, 2.).mean())

        if args.equator:
            print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(29.50 - 11.00) -
                                            10. * np.log10(np.power(sub_data - sub_inter, 2.).mean())))
        else:
            print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(1.93 + 30.35) -
                                            10. * np.log10(np.power(sub_data - sub_inter, 2.).mean())))

    if not args.equator:
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(1.93 + 30.35) -
                      10. * np.log10(mse / inter.shape[0]), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (1.93 + 30.35), max_diff.std() / (1.93 + 30.35)))
    else:
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(29.50 - 11.00) -
                      10. * np.log10(mse / inter.shape[0]), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (29.50 - 11.00), max_diff.std() / (29.50 - 11.00)))

if __name__ == "__main__":
    main(parse_args())
