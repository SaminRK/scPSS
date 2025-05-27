import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, ttest_ind, pearsonr

def mse(A, B):
    return np.mean((A[:, None, :] - B[None, :, :]) ** 2, axis=2)

def mae(A, B):
    return np.mean(np.abs(A[:, None, :] - B[None, :, :]), axis=2)

def energy_distance(A, B):
    d1 = cdist(A, A, metric='euclidean').mean()
    d2 = cdist(B, B, metric='euclidean').mean()
    d12 = cdist(A, B, metric='euclidean').mean()
    return 2 * d12 - d1 - d2  # scalar

def ks_test(A, B):
    assert A.shape[1] == B.shape[1] == 1, "KS test supports only 1D data"
    return np.array([[ks_2samp(a, b).statistic for b in B] for a in A])

def t_test(A, B):
    return np.array([[ttest_ind(a, b, equal_var=True).statistic for b in B] for a in A])

def pearson_distance(A, B):
    return np.array([[1 - pearsonr(a, b)[0] for b in B] for a in A])

def r2_distance(A, B):
    return np.array([[1 - (1 - np.sum((a - b)**2)/np.sum((a - np.mean(a))**2))
                      for b in B] for a in A])

def rbf_kernel(A, B, gamma=1.0):
    dists = cdist(A, B, 'sqeuclidean')
    return np.exp(-gamma * dists)

def mmd_rbf(A, B, gamma=1.0):
    """
    Computes MMD between two sample sets A and B using RBF kernel.
    A: shape (m, d)
    B: shape (n, d)
    """
    m, n = len(A), len(B)

    K_aa = rbf_kernel(A, A, gamma)
    K_bb = rbf_kernel(B, B, gamma)
    K_ab = rbf_kernel(A, B, gamma)

    # Remove diagonal for unbiased estimator
    np.fill_diagonal(K_aa, 0)
    np.fill_diagonal(K_bb, 0)

    mmd_sq = (K_aa.sum() / (m * (m - 1)) +
              K_bb.sum() / (n * (n - 1)) -
              2 * K_ab.sum() / (m * n))
    
    return mmd_sq  # scalar

custom_metrics = {
    "mse": mse,
    "mae": mae,
    "edistance": lambda A, B: np.full((A.shape[0], B.shape[0]), energy_distance(A, B)),
    "ks": ks_test,
    "t_test": t_test,
    "pearson": pearson_distance,
    "r2": r2_distance,
    "mmd": lambda A, B: np.full((A.shape[0], B.shape[0]), mmd_rbf(A, B))
}

CDIST_METRICS = {
    "braycurtis", "canberra", "chebyshev", "cityblock", "cosine",
    "euclidean", "mahalanobis", "minkowski", "seuclidean", "sqeuclidean"
}
