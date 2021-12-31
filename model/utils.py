import torch
from torch_sparse import spmm

import pdb

def batch_spmm(index, value, m, n, matrix):
    """
    :param index: (LongTensor) - The index tensor of sparse matrix.
    :param value: (Tensor) - The value tensor of sparse matrix.
    :param m: (int) - The first dimension of corresponding dense matrix.
    :param n: (int) - The second dimension of corresponding dense matrix.
    :param matrix: (Tensor) - Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    matrix = matrix.transpose(0, 1).reshape(n, -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    matrix = spmm(index, value, m, n, matrix)
    return matrix.reshape(m, batch_size, -1).transpose(1, 0)
