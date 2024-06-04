import torch

def gaussian_kernel(x, y, sigma=1.0):
    x_i = x[:, None, :]  # (M, 1, 4)
    y_j = y[None, :, :]  # (1, N, 4)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    return (-D_ij / (2 * sigma ** 2)).exp()  # (M, N) symbolic Gaussian kernel matrix

def laplacian_kernel(x, y, sigma=1.0):
    x_i = x[:, None, :]  # (M, 1, 1)
    y_j = y[None, :, :]  # (1, N, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    return (-D_ij.sqrt() / sigma).exp()  # (M, N) symbolic Laplacian kernel matrix