#!/usr/bin/env python

import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def truncated_svd(X, k):
    X_cent = X - np.mean(X, axis = 0, keepdims = True)
    X_cent /= np.sqrt(np.prod(X_cent.shape))
    U, S, Vt = np.linalg.svd(X_cent, full_matrices = False)
    U = U[:, 0:k]
    S = S[0:k]
    V = Vt.T
    F = V[:, 0:k]
    L = U @ np.diag(S)
    # F = V @ np.diag(S)
    S2 = np.square(S)
    return L, F, S2


def truncated_svd_fast(X, k):
    X_cent = X - np.mean(X, axis = 0, keepdims = True)
    tsvd = TruncatedSVD(n_components=k, n_iter=20)
    US = tsvd.fit_transform(csr_matrix(X_cent))
    Vt = tsvd.components_
    # F  = Vt.T @ np.diag(tsvd.singular_values_)
    F = Vt.T
    S2 = np.square(tsvd.singular_values_)
    return US, F, S2
