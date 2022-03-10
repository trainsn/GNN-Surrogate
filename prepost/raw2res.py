import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="raw2res")
parser.add_argument("--root", type=str, required=True,
                    help="root of the dataset")
parser.add_argument("--reference", type=str, default="0059_3.13326_1195.80632_0.59768_192.34201_temperature.npy",
                    help="root of the dataset")

args = parser.parse_args()

root = args.root
txt = "npyNames.txt"
train = False
if train:
    fh = open(os.path.join(root, "train", txt))
else:
    fh = open(os.path.join(root, "test", txt))

reference = np.load(os.path.join(root, "train", args.reference))
filenames = []
for filename in fh:
    filename = filename.strip("\r\n")
    if train:
        data_name = os.path.join(root, "train", filename)
    else:
        data_name = os.path.join(root, "test", filename)
    data = np.load(data_name)
    residual = data - reference
    print("{}, {:.6f}, {:.6f}, {:.6f}".format(filename, residual.min(), residual.max(), abs(residual).mean()))
    if train:
        np.save(os.path.join(root, "residual", "train", filename), residual)
        residual.tofile(os.path.join(root, "residual", "train", filename[:filename.rfind(".")] + ".bin"))
    else:
        np.save(os.path.join(root, "residual", "test", filename), residual)
        residual.tofile(os.path.join(root, "residual", "test", filename[:filename.rfind(".")] + ".bin"))
