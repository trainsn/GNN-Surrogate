import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from skimage import io

import pdb

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--equator", action="store_true", default=False,
                        help="compare the equator patch")
parser.add_argument("--rawRoot", type=str, default="/fs/ess/PAS0027/mpas_graph",
                        help="root of the raw data")
parser.add_argument("--reRoot", type=str, default="/fs/ess/PAS0027/MPAS1/Resample/",
                        help="root of the resampled data")
parser.add_argument("--rawCoord", type=str, default="/users/PAS0027/trainsn/mpas/mpas_graph/res/EC60to30/sphereCoord.npy",
                        help="path of raw data's coordinate")

if __name__ == '__main__':
    args = parser.parse_args()

    rawRoot = args.rawRoot
    fh = open(os.path.join(rawRoot, "test", "names.txt"))
    rawCoord = np.load(args.rawCoord)
    N = rawCoord.shape[0]

    reRoot = args.reRoot

    depth = np.unique(rawCoord[:, 2])
    depth[-1] += 1e-1
    depth_shape = depth.shape[0]
    lat_shape = 343
    lat = np.linspace(np.pi / 2, -np.pi / 2, lat_shape)
    lon_shape = 686
    lon = np.linspace(0., np.pi * 2, lon_shape)

    depth_lg = (np.tile(depth, (N, 1)) <= rawCoord[:, 2].reshape((N, 1))).sum(axis=1)
    depth_sm = depth_lg - 1
    depth_d = (rawCoord[:, 2] - depth[depth_sm]) / (depth[depth_lg] - depth[depth_sm])
    print("finish depth index calc")
    lat_sm = (np.tile(lat, (N, 1)) >= rawCoord[:, 0].reshape((N, 1))).sum(axis=1)
    lat_lg = lat_sm - 1
    lat_d = (rawCoord[:, 0] - lat[lat_sm]) / (lat[lat_lg] - lat[lat_sm])
    print("finish lat index calc")
    lon_lg = (np.tile(lon, (N, 1)) <= rawCoord[:, 1].reshape((N, 1))).sum(axis=1)
    lon_sm = lon_lg - 1
    lon_d = (rawCoord[:, 1] - lon[lon_sm]) / (lon[lon_lg] - lon[lon_sm])
    print("finish lon index calc")

    filenames = []
    for line in fh:
        filenames.append(line)

    if args.equator:
        equator = np.load(os.path.join(rawRoot, "equator_patch.npy"))
    mse = 0.
    psnrs = np.zeros(len(filenames))
    max_diff = np.zeros(len(filenames))
    dmin = -1.93
    dmax = 30.35

    for i in range(len(filenames)):
        filename = filenames[i].strip("\r\n")
        data = np.load(os.path.join(rawRoot, "test", filename))
        resample = np.zeros((depth_shape, lat_shape, lon_shape))
        for j in range(depth_shape):
            img = io.imread(os.path.join(reRoot, str(i + 70).zfill(4), "gray_layer" + str(j)  + ".png"))[:, :, 0]
            resample[j] = img / 255. * (dmax - dmin) + dmin
        c00 = (1 - depth_d) * resample[depth_sm, lat_sm, lon_sm] + depth_d * resample[depth_lg, lat_sm, lon_sm]
        c01 = (1 - depth_d) * resample[depth_sm, lat_sm, lon_lg] + depth_d * resample[depth_lg, lat_sm, lon_lg]
        c10 = (1 - depth_d) * resample[depth_sm, lat_lg, lon_sm] + depth_d * resample[depth_lg, lat_lg, lon_sm]
        c11 = (1 - depth_d) * resample[depth_sm, lat_lg, lon_lg] + depth_d * resample[depth_lg, lat_lg, lon_lg]
        c0 = (1 - lat_d) * c00 + lat_d * c10
        c1 = (1 - lat_d) * c01 + lat_d * c11
        fake_data = (1 - lon_d) * c0 + lon_d * c1

        if args.equator:
            data = data * equator
            data = data[abs(data) > 0]
            fake_data = fake_data * equator
            fake_data = fake_data[abs(fake_data) > 0]

        diff = abs(data - fake_data)
        max_diff[i] = diff.max()
        # print(diff.max())
        mse += np.power(data - fake_data, 2.).mean()
        # print(mse)
        if args.equator:
            psnrs[i] = 20. * np.log10(29.50 - 10.54) - 10. * np.log10(np.power(data - fake_data, 2.).mean())
        else:
            psnrs[i] = 20. * np.log10(1.93 + 30.35) - 10. * np.log10(np.power(data - fake_data, 2.).mean())

        if args.equator:
            print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(29.50 - 10.54) -
                                            10. * np.log10(np.power(data - fake_data, 2.).mean())))
        else:
            print("{:d} PSNR: {:4f}".format(i, 20. * np.log10(1.93 + 30.35) -
                                            10. * np.log10(np.power(data - fake_data, 2.).mean())))
        pdb.set_trace()

    if args.equator:
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(29.50 - 10.54) -
                      10. * np.log10(mse / len(filenames)), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (29.50 - 10.54), max_diff.std() / (29.50 - 10.54)))
    else:
        print("====> PSNR on raw avg {}, std var {}"
              .format(20. * np.log10(1.93 + 30.35) -
                      10. * np.log10(mse / len(filenames)), psnrs.std()))
        print("====> max difference on raw avg {}, std var {}"
              .format(max_diff.mean() / (1.93 + 30.35), max_diff.std() / (1.93 + 30.35)))
