import os
import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce, spspmm

import pdb

def load_graph(path):
    graphSizes = np.load(os.path.join(path, "ghtGraphSizes.npy"))
    valid = [1, 3, 5, 6, 8, 10, 11, 12]
    edgeNum = 7

    adjValues = []
    edgeOnes = []
    E_starts = []
    E_ends = []
    devices = [torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:0"),
               torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:0")]
    devices.insert(0, torch.device("cuda:0"))
    layer_id = 0
    for idx in valid:
        adjIdx = np.load(os.path.join(path, "ghtAdjacencyIdx" + str(idx) + ".npy"))
        adjValue0 = np.load(os.path.join(path, "ghtAdjacencyValue" + str(idx) + ".npy")).astype(np.float32)
        adjIdx = torch.from_numpy(adjIdx).to(devices[layer_id])
        adjValue0 = torch.from_numpy(adjValue0).to(devices[layer_id])

        edgeInfo = np.load(os.path.join(path, "ghtEdgeInfo" + str(idx) + ".npy")).astype(np.float32)
        edgeInfo = torch.from_numpy(edgeInfo).to(devices[layer_id])
        adjValue = []
        edgeOne = []
        E_start = []
        E_end = []

        for i in range(edgeNum):
            if i < 4:   # horizontal edges
                subEdge = edgeInfo[:, i] > 0.01
                adjValue.append((edgeInfo[subEdge, i] * adjValue0[subEdge]).type(torch.float16))
            else:
                subEdge = edgeInfo[:, i] > 0
                adjValue.append((edgeInfo[subEdge, i] * adjValue0[subEdge]).type(torch.float16))
            E = subEdge.sum().item()
            edgeIdx = torch.from_numpy(np.arange(0, E, 1)).type(torch.LongTensor).to(devices[layer_id])
            edgeOne.append(torch.ones(E).type(torch.float16).to(devices[layer_id]))
            E_start.append(torch.vstack((edgeIdx, adjIdx[0, subEdge])).to(devices[layer_id]))  # E x N (sparse)
            E_end.append(torch.vstack((edgeIdx, adjIdx[1, subEdge])).to(devices[layer_id]))  # E x N (sparse)
            del edgeIdx
        del edgeInfo
        del adjIdx
        del adjValue0

        adjValues.append(adjValue)
        edgeOnes.append(edgeOne)
        E_starts.append(E_start)
        E_ends.append(E_end)
        layer_id += 1

    avgPoolAsgnIndices = []
    avgPoolAsgnValues = []
    upAsgnIndices = []
    upAsgnValues = []

    for i in range(len(valid) - 1):
        idx = valid[i]
        avgPoolAsgnIdx = torch.from_numpy(np.load(os.path.join(path, "ghtAvgPoolAsgnIdx" + str(idx) + ".npy"))).type(torch.LongTensor).to(devices[layer_id])
        avgPoolAsgnValue = torch.from_numpy(np.load(os.path.join(path, "ghtAvgPoolAsgnValue" + str(idx) + ".npy")).astype(np.float32)).to(devices[layer_id])
        avgPoolAsgnValue = avgPoolAsgnValue.type(torch.float16)
        avgPoolAsgnIndices.append(avgPoolAsgnIdx)
        avgPoolAsgnValues.append(avgPoolAsgnValue)

    for i in range(1, len(valid)):
        idx = valid[i]
        upAsgnIdx = torch.from_numpy(np.load(os.path.join(path, "ghtUpAsgnIdx" + str(idx) + ".npy"))).type(torch.LongTensor).to(devices[layer_id])
        upAsgnValue = torch.from_numpy(np.load(os.path.join(path, "ghtUpAsgnValue" + str(idx) + ".npy")).astype(np.float32)).to(devices[layer_id])
        upAsgnValue = upAsgnValue.type(torch.float16)
        upAsgnIndices.append(upAsgnIdx)
        upAsgnValues.append(upAsgnValue)

    torch.cuda.empty_cache()
    return graphSizes[valid], adjValues, edgeOnes, E_starts, E_ends, avgPoolAsgnIndices, avgPoolAsgnValues, upAsgnIndices, upAsgnValues
