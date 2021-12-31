import os
import numpy as np
import torch
from torch_sparse import spmm

import pdb

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--root", type=str, default="/fs/project/PAS0027/mpas_graph",
                    help="root of the dataset")
parser.add_argument("--ght", type=str, default="ght_0.5",
                    help="path of the graph hierarchy")

args = parser.parse_args()

path = os.path.join(args.root, args.ght, "graph")
avgPoolAsgnIdx = torch.from_numpy(np.load(os.path.join(path, "ghtAvgPoolAsgnIdx0.npy"))).type(torch.LongTensor).to('cuda:0')
avgPoolAsgnValue = torch.from_numpy(np.load(os.path.join(path, "ghtAvgPoolAsgnValue0.npy")).astype(np.float32)).to('cuda:0')
upAsgnIdx = torch.from_numpy(np.load(os.path.join(path, "ghtUpAsgnIdx1.npy"))).type(torch.LongTensor).to('cuda:0')
upAsgnValue = torch.from_numpy(np.load(os.path.join(path, "ghtUpAsgnValue1.npy")).astype(np.float32)).to('cuda:0')

root = args.root
txt = "npyNames.txt"
for train in [True, False]:
    if train:
        fh = open(os.path.join(root, "residual", "train", txt))
    else:
        fh = open(os.path.join(root, "residual", "test", txt))

    filenames = []
    for filename in fh:
        filename = filename.strip("\r\n")
        if train:
            data_name = os.path.join(root, "residual", "train", filename)
        else:
            data_name = os.path.join(root, "residual", "test", filename)
        raw = torch.from_numpy(np.load(data_name).astype(np.float32)).to('cuda:0')

        m = avgPoolAsgnIdx[0].max() + 1
        n = avgPoolAsgnIdx[1].max() + 1

        ght = spmm(avgPoolAsgnIdx, avgPoolAsgnValue, m, n, raw)
        ght = torch.squeeze(ght).cpu().numpy().astype(np.float64)

        if train:
            np.save(os.path.join(root, args.ght, "train", filename), ght)
            ght.tofile(os.path.join(root, args.ght, "train", filename[:filename.rfind(".")] + ".bin"))
        else:
            np.save(os.path.join(root, args.ght, "test", filename), ght)
            ght.tofile(os.path.join(root, args.ght, "test", filename[:filename.rfind(".")] + ".bin"))

# m = upAsgnIdx[0].max() + 1
# n = upAsgnIdx[1].max() + 1
# recon = spmm(upAsgnIdx, upAsgnValue, m, n, ght)
# recon = torch.squeeze(recon)
# diff = raw - recon
# recon = recon.cpu().numpy().astype(np.float64)
# np.save(filename + "_ght" + ".npy", recon)
# recon.tofile(filename + "_ght" + ".bin")
