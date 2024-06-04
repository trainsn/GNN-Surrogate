import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from skimage import io

import pdb

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--rawRoot", type=str, default="/fs/ess/PAS0027/mpas_graph",
                        help="root of the raw data")
parser.add_argument("--reRoot", type=str, default="/fs/ess/PAS0027/MPAS1/Resample/",
                        help="root of the resampled data")


if __name__ == '__main__':
    args = parser.parse_args()

    rawRoot = args.rawRoot
    fh = open(os.path.join(rawRoot, "train", "names.txt"))
    reRoot = args.reRoot
    
    filenames = []
    for line in fh:
        filenames.append(line)
        
    dmin = -1.93
    dmax = 30.35
    
    lat_shape = 340
    lon_shape = 680
    depth_shape = 60
    
    
    for i in range(len(filenames)):
        resample = np.zeros((depth_shape, lat_shape, lon_shape), dtype=np.float32)
        for j in range(depth_shape):
            img = io.imread(os.path.join(reRoot, str(i).zfill(4), "layer" + str(j)  + ".png"))
            resample[j] = img / 255. * (dmax - dmin) + dmin
    
        resample.tofile(os.path.join(reRoot, filenames[i][:filenames[i].rfind('.')] + ".raw"))
        pdb.set_trace()
            