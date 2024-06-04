import os
import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.metrics.pairwise import euclidean_distances
import pyemd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm


import pdb

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
parser.add_argument("--dir", type=str, default="Pred",
                        help="Pred or Inter")
parser.add_argument("--mode", required=True, type=str,
                        help="map, equator, lat, lon, iso, depth")


def compute_emd(im1, im2, cost_mat, l_bins=8, a_bins=12, b_bins=12):
    lab_im1 = color.rgb2lab(im1.astype(np.uint8))
    lab_im1 = lab_im1.reshape((lab_im1.shape[0] * lab_im1.shape[1], lab_im1.shape[2]))
    lab_hist_1, _ = np.histogramdd(lab_im1, bins=(l_bins, a_bins, b_bins), range=[[0., 100.], [-86.185, 98.254], [-107.863, 94.482]], normed=False)

    lab_im2 = color.rgb2lab(im2.astype(np.uint8))
    lab_im2 = lab_im2.reshape((lab_im2.shape[0] * lab_im2.shape[1], lab_im2.shape[2]))
    lab_hist_2, _ = np.histogramdd(lab_im2, bins=(l_bins, a_bins, b_bins), range=[[0., 100.], [-86.185, 98.254], [-107.863, 94.482]], normed=False)

    n_bins = l_bins * a_bins * b_bins
    lab_hist_1 = lab_hist_1.reshape((n_bins))
    lab_hist_2 = lab_hist_2.reshape((n_bins))
    img_res = lab_im1.shape[0]
    lab_hist_1 /= img_res
    lab_hist_2 /= img_res
    return pyemd.emd(lab_hist_1, lab_hist_2, cost_mat)


def compute_emd_cost_mat(l_bins=8, a_bins=12, b_bins=12):
    n_bins = l_bins * a_bins * b_bins
    index_mat = np.zeros((l_bins, a_bins, b_bins, 3))
    for idx in range(l_bins):
        for jdx in range(a_bins):
            for kdx in range(b_bins):
                index_mat[idx, jdx, kdx] = np.array([idx, jdx, kdx])
    index_mat = index_mat.reshape(n_bins, 3)
    all_dists = euclidean_distances(index_mat, index_mat)
    return all_dists / np.max(all_dists)

if __name__ == '__main__':
    emd_cost_mat = compute_emd_cost_mat()

    args = parser.parse_args()
    print(args)
    root1 = os.path.join(args.root, "Results")
    root2 = os.path.join(args.root, args.dir)

    num_data = 30
    if args.mode == "map":
        num_layers = 60
        ssims = np.zeros((num_layers, num_data)).astype(np.float64)
        emds = np.zeros((num_layers, num_data)).astype(np.float64)
        for i in tqdm(range(70, 100)):
            dire = str(i).zfill(4)
            for j in range(0, num_layers):
                img1 = io.imread(os.path.join(root1, dire, "layer" + str(j) + ".png"))
                img2 = io.imread(os.path.join(root2, dire, "layer" + str(j) + ".png"))
                ssim = compute_ssim(img1, img2, data_range=255., multichannel=True)
                ssims[j, i - 70] = ssim
                emd = compute_emd(img1, img2, emd_cost_mat)
                emds[j, i - 70] = emd
        for j in range(0, num_layers):
            print("layer{:d}: SSIM {:4f} ".format(j, ssims[j].mean()))
        for j in range(0, num_layers):
            print("layer{:d}: EMD {:4f}".format(j, emds[j].mean()))
        # for j in range(0, num_layers):
        #     print("layer{:d}: SSIM sd {:4f}".format(j, ssims[j].std()))
        # for j in range(0, num_layers):
        #     print("layer{:d}: EMD sd {:4f}".format(j, emds[j].std()))
    elif args.mode == "equator":
        num_layers = 20
        ssims = np.zeros((num_layers, num_data)).astype(np.float64)
        emds = np.zeros((num_layers, num_data)).astype(np.float64)
        for i in tqdm(range(70, 100)):
            dire = "equator" + str(i).zfill(4)
            for j in range(0, num_layers):
                img1 = io.imread(os.path.join(root1, dire, "layer" + str(j) + ".png"))
                img2 = io.imread(os.path.join(root2, dire, "layer" + str(j) + ".png"))
                ssim = compute_ssim(img1, img2, data_range=255., multichannel=True)
                ssims[j, i - 70] = ssim
                emd = compute_emd(img1, img2, emd_cost_mat)
                emds[j, i - 70] = emd
        for j in range(0, num_layers):
            print("layer{:d}: SSIM avg {:4f}".format(j, ssims[j].mean()))
        for j in range(0, num_layers):
            print("layer{:d}: EMD avg {:4f}".format(j, emds[j].mean()))
        # for j in range(0, num_layers):
        #     print("layer{:d}: SSIM sd {:4f}".format(j, ssims[j].std()))
        # for j in range(0, num_layers):
        #     print("layer{:d}: EMD sd {:4f}".format(j, emds[j].std()))
    elif args.mode == "lat":
        num_layers = 9
        ssims = np.zeros((num_layers, num_data)).astype(np.float64)
        emds = np.zeros((num_layers, num_data)).astype(np.float64)
        for i in tqdm(range(70, 100)):
            dire = str(i).zfill(4)
            for j in range(0, num_layers):
                img1 = io.imread(os.path.join(root1, dire, "lat" + str(-60 + 15 * j) + ".png"))
                img2 = io.imread(os.path.join(root2, dire, "lat" + str(-60 + 15 * j) + ".png"))
                ssim = compute_ssim(img1, img2, data_range=255., multichannel=True)
                ssims[j, i - 70] = ssim
                emd = compute_emd(img1, img2, emd_cost_mat)
                emds[j, i - 70] = emd
        for j in range(0, num_layers):
            print("lat{:d}: SSIM {:4f}".format(-60 + 15 * j, ssims[j].mean()))
        for j in range(0, num_layers):
            print("lat{:d}: EMD {:4f}".format(-60 + 15 * j, emds[j].mean()))
        # for j in range(0, num_layers):
        #     print("lat{:d}: SSIM sd {:4f}".format(-60 + 15 * j, ssims[j].std()))
        # for j in range(0, num_layers):
        #     print("lat{:d}: EMD sd {:4f}".format(-60 + 15 * j, emds[j].std()))
    elif args.mode == "lon":
        num_layers = 12
        ssims = np.zeros((num_layers, num_data)).astype(np.float64)
        emds = np.zeros((num_layers, num_data)).astype(np.float64)
        for i in tqdm(range(70, 100)):
            dire = str(i).zfill(4)
            for j in range(0, num_layers):
                img1 = io.imread(os.path.join(root1, dire, "lon" + str(15 + 30 * j) + ".png"))
                img2 = io.imread(os.path.join(root2, dire, "lon" + str(15 + 30 * j) + ".png"))
                ssim = compute_ssim(img1, img2, data_range=255., multichannel=True)
                ssims[j, i - 70] = ssim
                emd = compute_emd(img1, img2, emd_cost_mat)
                emds[j, i - 70] = emd
        for j in range(0, num_layers):
            print("lon{:d}: SSIM {:4f}".format(15 + 30 * j, ssims[j].mean()))
        for j in range(0, num_layers):
            print("lon{:d}: EMD {:4f}".format(15 + 30 * j, emds[j].mean()))
        # for j in range(0, num_layers):
        #     print("lon{:d}: SSIM sd {:4f}".format(15 + 30 * j, ssims[j].std()))
        # for j in range(0, num_layers):
        #     print("lon{:d}: EMD sd {:4f}".format(15 + 30 * j, emds[j].std()))
    elif args.mode == "iso":
        num_layers = 50
        ssim = 0.
        emd = 0.
        for i in tqdm(range(70, 100)):
            dire = str(i).zfill(4)
            for j in range(0, num_layers):
                img1 = io.imread(os.path.join(root1, dire, "iso" + str(j) + ".png"))
                img2 = io.imread(os.path.join(root2, dire, "iso" + str(j) + ".png"))
                ssim += compute_ssim(img1, img2, data_range=255., multichannel=True)
                emd += compute_emd(img1, img2, emd_cost_mat)
        print("lat{:d}: SSIM {:4f}".format(j, ssim / 30.0 / num_layers))
        print("lat{:d}: EMD {:4f}".format(j, emd / 30.0 / num_layers))
    elif args.mode == "depth":
        num_layers = 11
        dists = np.zeros((num_layers, num_data)).astype(np.float64)
        dices = np.zeros((num_layers, num_data)).astype(np.float64)
        for i in tqdm(range(70, 100)):
            dire = str(i).zfill(4)
            for j in range(0, num_layers):
                d1 = np.load(os.path.join(root1, dire, "depth" + str(25 - j * 2) + ".npy"))
                d2 = np.load(os.path.join(root2, dire, "depth" + str(25 - j * 2) + ".npy"))
                dists[j, i - 70] = abs(d1[(d1 * d2) > 0] - d2[(d1 * d2) > 0]).mean()
                dices[j, i - 70] = ((d1 * d2) > 0).sum() / ((d1 + d2) > 0).sum()
        for j in range(0, num_layers):
            print("isovalue{:d}: dist {:4f}".format(25 - j * 2, dists[j].mean()))
        for j in range(0, num_layers):
            print("isovalue{:d}: Jaccard {:4f}".format(25 - j * 2, dices[j].mean()))
        # for j in range(0, num_layers):
        #     print("isovalue{:d}: dist SD {:4f}".format(25 - j * 2, dists[j].std()))
        # for j in range(0, num_layers):
        #     print("isovalue{:d}: Jaccard SD {:4f}".format(25 - j * 2, dices[j].std()))
    elif args.mode == "depth-equator":
        equator = np.load(os.path.join(args.root, "equator_patch.npy"))
        num_layers = 11
        dists = np.zeros((num_layers, num_data)).astype(np.float64)
        dices = np.zeros((num_layers, num_data)).astype(np.float64)
        for i in tqdm(range(70, 100)):
            dire = str(i).zfill(4)
            for j in range(0, num_layers):
                d1 = np.load(os.path.join(root1, dire, "depth" + str(25 - j * 2) + ".npy"))
                d2 = np.load(os.path.join(root2, dire, "depth" + str(25 - j * 2) + ".npy"))
                d1 = d1 * equator
                d2 = d2 * equator
                dists[j, i - 70] = abs(d1[(d1 * d2) > 0] - d2[(d1 * d2) > 0]).mean()
                dices[j, i - 70] = ((d1 * d2) > 0).sum() / ((d1 + d2) > 0).sum()
        for j in range(0, num_layers):
            print("isovalue{:d}: dist {:4f}".format(25 - j * 2, dists[j].mean()))
        for j in range(0, num_layers):
            print("isovalue{:d}: Jaccard {:4f}".format(25 - j * 2, dices[j].mean()))
        # for j in range(0, num_layers):
        #     print("isovalue{:d}: dist SD {:4f}".format(25 - j * 2, dists[j].std()))
        # for j in range(0, num_layers):
        #     print("isovalue{:d}: Jaccard SD {:4f}".format(25 - j * 2, dices[j].std()))




