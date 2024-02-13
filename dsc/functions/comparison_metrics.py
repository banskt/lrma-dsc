#!/usr/bin/env python

import numpy as np
from scipy.spatial import procrustes
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as skmetrics


def mean_squared_error(original, recovered, mask = None):
    if mask is None: mask = np.ones_like(original)
    n = np.sum(mask)
    mse = np.sum(np.square((original - recovered) * mask)) / n
    return mse


def peak_signal_to_noise_ratio(original, recovered, mask = None):
    if mask is None: mask = np.ones_like(original)
    omax = np.max(original[mask == 1])
    omin = np.min(original[mask == 1])
    maxsig2 = np.square(omax - omin)
    mse = mean_squared_error(original, recovered, mask)
    res = 10 * np.log10(maxsig2 / mse)
    return res


def matrix_dissimilarity_scores(original, recovered, mask = None, match = 'zerofill'):
    '''
    Procrustes analysis returns the square of the Frobenius norm.
    Use the rotated matrix to obtain the peak signal-to-noise ratio (PSNR).
    Input matrices can have different dimensions.
    There are two ways to match:
        - clip: remove information from the larger matrix
        - zerofill: pad zero columns in the smaller matrix
    '''
    n_orig = original.shape[1]
    n_recv = recovered.shape[1]
    m = original.shape[0]
    if match == 'clip':
        n = min(n_orig, n_recv)
        X = original[:, :n]
        Y = recovered[:, :n]
    elif match == 'zerofill':
        n = max(n_orig, n_recv)
        X = np.zeros((m, n))
        Y = np.zeros((m, n))
        X[:, :n_orig] = original
        Y[:, :n_recv] = recovered
    R_orig, R_recv, m2 = procrustes(X, Y)
    psnr = peak_signal_to_noise_ratio(R_orig, R_recv, mask)
    return np.sqrt(m2), psnr


def adjusted_mutual_information_score(X, class_labels):
    X_cent = X - np.mean(X, axis = 0, keepdims = True)
    distance_matrix = skmetrics.pairwise.pairwise_distances(X_cent, metric='euclidean')
    model = AgglomerativeClustering(n_clusters = 5, linkage = 'average', metric = 'precomputed')
    class_pred = model.fit_predict(distance_matrix)
    return skmetrics.adjusted_mutual_info_score(class_labels, class_pred)
