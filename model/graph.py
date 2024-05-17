import os
import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce, spspmm

import pdb

def load_graph(path):
    graphSizes = np.load(os.path.join(path, "graphSizes.npy"))
    valid = [0, 2, 4, 5, 7, 9, 10, 11]
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
        adjIdx = np.load(os.path.join(path, "adjacencyIdx" + str(idx) + ".npy"))
        adjValue0 = np.load(os.path.join(path, "adjacencyValue" + str(idx) + ".npy")).astype(np.float32)
        adjIdx = torch.from_numpy(adjIdx).to(devices[layer_id])
        adjValue0 = torch.from_numpy(adjValue0).to(devices[layer_id])

        edgeInfo = np.load(os.path.join(path, "edgeInfo" + str(idx) + ".npy")).astype(np.float32)
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
    for i in range(len(valid) - 1):
        idx = valid[i]
        if valid[i+1] - valid[i] == 2:
            avgPoolAsgnIdxA = torch.from_numpy(np.load(os.path.join(path, "avgPoolAsgnIdx" + str(idx) + ".npy"))).type(torch.LongTensor).to('cuda:0')
            avgPoolAsgnValueA = torch.from_numpy(np.load(os.path.join(path, "avgPoolAsgnValue" + str(idx) + ".npy")).astype(np.float32)).to('cuda:0')
            avgPoolAsgnIdxB = torch.from_numpy(np.load(os.path.join(path, "avgPoolAsgnIdx" + str(idx+1) + ".npy"))).type(torch.LongTensor).to('cuda:0')
            avgPoolAsgnValueB = torch.from_numpy(np.load(os.path.join(path, "avgPoolAsgnValue" + str(idx+1) + ".npy")).astype(np.float32)).to('cuda:0')
            avgPoolAsgnIdx, avgPoolAsgnValue = spspmm(avgPoolAsgnIdxB, avgPoolAsgnValueB, avgPoolAsgnIdxA, avgPoolAsgnValueA,
                                                      graphSizes[idx+2], graphSizes[idx+1], graphSizes[idx])
            avgPoolAsgnIdx = avgPoolAsgnIdx.to('cpu')
            avgPoolAsgnValue = avgPoolAsgnValue.to('cpu').type(torch.float16)
        else:
            avgPoolAsgnIdx = torch.from_numpy(np.load(os.path.join(path, "avgPoolAsgnIdx" + str(idx) + ".npy"))).type(torch.LongTensor)
            avgPoolAsgnValue = torch.from_numpy(np.load(os.path.join(path, "avgPoolAsgnValue" + str(idx) + ".npy")).astype(np.float16))
        avgPoolAsgnIndices.append(avgPoolAsgnIdx)
        avgPoolAsgnValues.append(avgPoolAsgnValue)

    upAsgnIndices = []
    upAsgnValues = []
    for i in range(1, len(valid)):
        idx = valid[i]
        if valid[i] - valid[i-1] == 2:
            upAsgnIdxA = torch.from_numpy(np.load(os.path.join(path, "upAsgnIdx" + str(idx) + ".npy"))).type(torch.LongTensor).to('cuda:0')
            upAsgnValueA = torch.from_numpy(np.load(os.path.join(path, "upAsgnValue" + str(idx) + ".npy")).astype(np.float32)).to('cuda:0')
            upAsgnIdxB = torch.from_numpy(np.load(os.path.join(path, "upAsgnIdx" + str(idx-1) + ".npy"))).type(torch.LongTensor).to('cuda:0')
            upAsgnValueB = torch.from_numpy(np.load(os.path.join(path, "upAsgnValue" + str(idx-1) + ".npy")).astype(np.float32)).to('cuda:0')
            upAsgnIdx, upAsgnValue = spspmm(upAsgnIdxB, upAsgnValueB, upAsgnIdxA, upAsgnValueA,
                                            graphSizes[idx-2], graphSizes[idx-1], graphSizes[idx])
            upAsgnIdx = upAsgnIdx.to('cpu')
            upAsgnValue = upAsgnValue.to('cpu').type(torch.float16)
        else:
            upAsgnIdx = torch.from_numpy(np.load(os.path.join(path, "upAsgnIdx" + str(idx) + ".npy"))).type(torch.LongTensor)
            upAsgnValue = torch.from_numpy(np.load(os.path.join(path, "upAsgnValue" + str(idx) + ".npy")).astype(np.float16))
        upAsgnIndices.append(upAsgnIdx)
        upAsgnValues.append(upAsgnValue)

    torch.cuda.empty_cache()
    return graphSizes[valid], adjValues, edgeOnes, E_starts, E_ends, avgPoolAsgnIndices, avgPoolAsgnValues, upAsgnIndices, upAsgnValues
