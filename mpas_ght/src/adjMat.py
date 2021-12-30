import os
import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce, spspmm

import pdb

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

root = "EC60to30_0.75"
graphSizes = np.load(os.path.join(root, "ghtGraphSizes.npy"))
path = "/fs/project/PAS0027/mpas_graph/ght_0.75/graph"

for idx in range(0, graphSizes.shape[0] - 1):
    idxFile = "ghtAdjacencyIdx" + str(idx) + ".npy"
    valueFile = "ghtAdjacencyValue" + str(idx) + ".npy"
    edgeFile = "ghtEdgeInfo" + str(idx) + ".npy"
    edgeIdx0 = np.load(os.path.join(root, idxFile))
    edgeInfo0 = np.load(os.path.join(root, valueFile)).astype(np.float32)
    N = edgeIdx0.max() + 1

    # horizontal weight
    for i in range(4):
        edgeInfo0[i] = edgeInfo0[i] * edgeInfo0[4] * edgeInfo0[-1]
    # vertical weight
    edgeInfo0[4] = edgeInfo0[5] * edgeInfo0[-1]
    edgeInfo0[5] = edgeInfo0[6] * edgeInfo0[-1]
    # now use as the dimension for self edges
    edgeInfo0[6] = np.zeros(edgeInfo0.shape[1])
    edgeInfo0 = edgeInfo0[:-1]

    edgeIdx0 = torch.from_numpy(edgeIdx0.astype(np.int64))
    edgeInfo0 = torch.from_numpy(edgeInfo0.T)
    edgeIdx0, edgeInfo0 = coalesce(edgeIdx0, edgeInfo0, m=graphSizes[idx], n=graphSizes[idx], op="mean")
    edgeIdx0 = edgeIdx0.numpy()
    edgeInfo0 = edgeInfo0.numpy().T

    adj = sp.csr_matrix((np.ones(edgeInfo0.shape[1]), (edgeIdx0[0], edgeIdx0[1])), shape=(graphSizes[idx], graphSizes[idx]))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj.tocoo().astype(np.float32)
    adjIdx = np.vstack((adj.row, adj.col)).astype(np.int64)
    adjValue = adj.data
    adjIdx = torch.from_numpy(adjIdx)
    adjValue = torch.from_numpy(adjValue)
    adjIdx, adjValue = coalesce(adjIdx, adjValue, m=graphSizes[idx], n=graphSizes[idx])

    edgeIdxSelf = np.tile(np.arange(0, N, 1), (2, 1))
    edgeInfoSelf = np.concatenate((np.zeros((edgeInfo0.shape[0] - 1, N)), np.ones((1, N))), axis=0)

    edgeIdx = np.concatenate((edgeIdx0, edgeIdxSelf), axis=1)
    edgeInfo = np.concatenate((edgeInfo0, edgeInfoSelf), axis=1)
    edgeIdx = torch.from_numpy(edgeIdx)
    edgeInfo = torch.from_numpy(edgeInfo.T)
    edgeIdx, edgeInfo = coalesce(edgeIdx, edgeInfo, m=graphSizes[idx], n=graphSizes[idx])
    assert (adjIdx != edgeIdx).sum() == 0

    np.save(os.path.join(path, idxFile), adjIdx.type(torch.int32))
    np.save(os.path.join(path, valueFile), adjValue.type(torch.float16))
    np.save(os.path.join(path, edgeFile), edgeInfo.type(torch.float16))
