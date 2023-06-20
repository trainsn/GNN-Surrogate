import os
import argparse
import numpy as np
from tqdm import tqdm

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Neareset Neighbor")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--num-run", type=int, default=0,
                        help="the number of Ensemble Runs")
    parser.add_argument("--g", type=int, default=3,
                        help="the number of Ensemble Runs")
    parser.add_argument("--equator", action="store_true", default=False,
                        help="compare the equator patch")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save the npy file")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    root = args.root
    num_run = args.num_run
    g = args.g
    test_params = np.load(os.path.join(root, "test/params.npy"))
    if num_run > 0:
        train_params = np.load(os.path.join(root, "train/params_" + str(num_run) + ".npy"))
    else:
        train_params = np.load(os.path.join(root, "train/params.npy"))

    BwsAMin, BwsAMax = 0.0, 5.0
    kappaMin, kappaMax = 300.0, 1500.0
    cvmixMin, cvmixMax = 0.25, 1.0
    momMin, momMax = 100.0, 300.0

    test_params[:, 1] = (test_params[:, 1] - BwsAMin) / (BwsAMax - BwsAMin)
    test_params[:, 2] = (test_params[:, 2] - kappaMin) / (kappaMax - kappaMin)
    test_params[:, 3] = (test_params[:, 3] - cvmixMin) / (cvmixMax - cvmixMin)
    test_params[:, 4] = (test_params[:, 4] - momMin) / (momMax - momMin)

    train_params[:, 1] = (train_params[:, 1] - BwsAMin) / (BwsAMax - BwsAMin)
    train_params[:, 2] = (train_params[:, 2] - kappaMin) / (kappaMax - kappaMin)
    train_params[:, 3] = (train_params[:, 3] - cvmixMin) / (cvmixMax - cvmixMin)
    train_params[:, 4] = (train_params[:, 4] - momMin) / (momMax - momMin)

    if num_run > 0:
        fh = open(os.path.join(root, "train", "npyNames_" + str(num_run) + ".txt"))
    else:
        fh = open(os.path.join(root, "train", "npyNames.txt"))
    train_filenames = []
    for line in fh:
        train_filenames.append(line)

    fh = open(os.path.join(root, "test", "npyNames.txt"))
    test_filenames = []
    for line in fh:
        test_filenames.append(line)

    equator = np.load(os.path.join(root, "equator_patch.npy"))

    mse = 0.
    psnrs = np.zeros(len(test_filenames))
    max_diff = np.zeros(len(test_filenames))
    for i in tqdm(range(test_params.shape[0])):
        param = test_params[i][1:]
        filename = test_filenames[i].strip("\r\n")
        data = np.load(os.path.join(root, "test", filename))

        param_dist = abs(param - train_params[:, 1:]).sum(axis=1)
        indices = np.argpartition(param_dist, g)

        sum_data = np.zeros_like(data)
        sum_dist = 0.0
        for j in range(g):
            train_filename = train_filenames[indices[j]].strip("\r\n")
            train_data = np.load(os.path.join(root, "train", train_filename))
            sum_data += 1. / param_dist[indices[j]] * train_data
            sum_dist += 1. / param_dist[indices[j]]
        inter = sum_data / sum_dist

        if args.equator:
            data = data * equator
            data = data[data > 0]
            inter = inter * equator
            inter = inter[inter > 0]
        diff = abs(data - inter)
        max_diff[i] = diff.max()
        mse += np.power(data - inter, 2.).mean()
        if args.equator:
            psnrs[i] = 20. * np.log10(29.50 - 11.00) - 10. * np.log10(np.power(data - inter, 2.).mean())
        else:
            psnrs[i] = 20. * np.log10(1.93 + 30.35) - 10. * np.log10(np.power(data - inter, 2.).mean())

        if args.save:
            np.save(os.path.join(args.root, "test", "{:04d}_temperature_inter.npy".format(
                int(test_params[i][0])
            )), inter)
            inter.tofile(os.path.join(args.root, "test", "{:04d}_temperature_inter.bin".format(
                int(test_params[i][0])
            )))
        if args.equator:
            print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(29.50 - 11.00) -
                                            10. * np.log10(np.power(data - inter, 2.).mean())))
        else:
            print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(1.93 + 30.35) -
                                            10. * np.log10(np.power(data - inter, 2.).mean())))


    if args.equator:
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(29.50 - 11.00) -
                      10. * np.log10(mse / len(test_filenames)), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (29.50 - 11.00), max_diff.std() / (29.50 - 11.00)))
    else:
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(1.93 + 30.35) -
                      10. * np.log10(mse / len(test_filenames)), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (1.93 + 30.35), max_diff.std() / (1.93 + 30.35)))


if __name__ == "__main__":
  main(parse_args())