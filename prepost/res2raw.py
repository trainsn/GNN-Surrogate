import os
import argparse
import numpy as np
import pdb

parser = argparse.ArgumentParser(description="res2raw")
parser.add_argument("--root", type=str, required=True,
                    help="root of the dataset")
parser.add_argument("--reference", type=str, default="0059_3.13326_1195.80632_0.59768_192.34201_temperature.npy",
                    help="root of the dataset")
parser.add_argument("--ght", type=str, default="ght_0.5",
                    help="path of the graph hierarchy")
args = parser.parse_args()

root = args.root

reference = np.load(os.path.join(root, "train", args.reference))
for i in range(70, 100):
    filename = str(i).zfill(4) + "_temperature_fake.npy"
    fake_residual = np.load(os.path.join(root, args.ght, "test", filename))
    fake_data = fake_residual + reference
    np.save(os.path.join(root, "test", filename), fake_data)
    fake_data.tofile(os.path.join(root, "test", filename[:filename.rfind(".")] + ".bin"))
