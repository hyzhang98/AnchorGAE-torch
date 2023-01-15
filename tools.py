import torch
import numpy as np

def distance2(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result

def getB_via_CAN(distances, k):
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, k]
    top_k = torch.t(top_k.repeat(distances.shape[1], 1)) + 10 ** -10
    sum_top_k = torch.sum(sorted_distances[:, 0:k], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(distances.shape[1], 1))
    weights = torch.div(top_k - distances, k * top_k - sum_top_k + 1e-7)
    weights = weights.relu()
    return weights

def recons_c2(m, B, embedding, embedding_dim):
    f1 = embedding.t().matmul(B)
    Bsum = B.sum(dim=0).repeat(embedding_dim, 1)
    return (f1/Bsum + 1e-7).t()

def reconstruct_B(centroids, embedding, num_neighbors):
    f = distance2(embedding.t(), centroids.t(), square=True)
    f = torch.tensor(f)
    B = getB_via_CAN(f, num_neighbors)
    return B