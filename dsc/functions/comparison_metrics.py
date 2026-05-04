#!/usr/bin/env python

import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as skmetrics


def standardize(X, axis = 0, center = True, scale = True):
    if center:
        X = X - np.mean(X, axis = axis, keepdims = True)
    if scale:
        std = np.std(X, axis = axis, keepdims = True)
        std_safe = np.where(std == 0, 1.0, std)  # avoid division by zero
        X = X / std_safe
    return X


def mean_squared_error(original, recovered, mask = None):
    if mask is None: mask = np.ones_like(original)
    n = np.sum(mask)
    mse = np.sum(np.square((original - recovered) * mask)) / n
    return mse


def root_mean_squared_error(original, recovered, mask = None):
    mse = mean_squared_error(original, recovered, mask = mask)
    return np.sqrt(mse)


def relative_rmse(rmse, original):
    """
    This is the same as relative Frobenius error.
    ||A - B||_F / ||A||_F
    """
    return rmse / np.sqrt(np.mean(original ** 2))


def global_calibration(Z, Z_hat, eps = 1e-12):
    denom = np.sum(Z_hat ** 2)
    true_norm_sq = np.sum(Z ** 2)
    threshold = eps**2 * true_norm_sq
    if denom <= threshold:
        sg = 0.0
        is_zero_norm = True
    else:
        sg = np.sum(Z * Z_hat) / denom
        is_zero_norm = False
    return sg, is_zero_norm


def peak_signal_to_noise_ratio(original, recovered, mask = None):
    if mask is None: mask = np.ones_like(original)
    omax = np.max(original[mask == 1])
    omin = np.min(original[mask == 1])
    maxsig2 = np.square(omax - omin)
    mse = mean_squared_error(original, recovered, mask)
    res = 10 * np.log10(maxsig2 / mse)
    return res


def match_latent_dimensions(original, recovered, dim_policy = "zerofill"):
    """
    Ensure both matrices use the same latent dimension K.

    L : (n_traits, K)
    F : (n_variants, K)

    dim_policy:
      - 'clip'     : keep only min(K_true, K_hat)
      - 'zerofill' : pad the smaller one with zero columns
    """
    k_true = original.shape[1]
    k_hat = recovered.shape[1]

    if dim_policy == "clip":
        k = min(k_true, k_hat)
        X = original[:, :k]
        Y = recovered[:, :k]
        return (X, Y)

    if dim_policy == "zerofill":

        def pad_cols(X, k_out):
            out = np.zeros((X.shape[0], k_out))
            out[:, :X.shape[1]] = X
            return out

        k = max(k_true, k_hat)
        X = pad_cols(original, k)
        Y = pad_cols(recovered, k)
        return (X, Y)

    raise ValueError("dim_policy must be 'clip' or 'zerofill'")


def matrix_dissimilarity_scores(original, recovered, mask = None, dim_policy = "zerofill"):
    '''
    Procrustes analysis returns the square of the Frobenius norm.
    Use the rotated matrix to obtain the peak signal-to-noise ratio (PSNR).
    Input matrices can have different dimensions.
    There are two ways to match dimensions using `dim_policy`:
        - clip: remove information from the larger matrix
        - zerofill: pad zero columns in the smaller matrix
    '''
    X, Y = match_latent_dimensions(original, recovered, dim_policy = dim_policy)

    # gleanr sometimes produces a single column of zero values
    # procrustes requires: Input matrices must contain >1 unique points
    # a matrix has no unique points iff max - min == 0.
    if np.ptp(Y) == 0 :
        m2 = np.sum(np.square(X))
        psnr = peak_signal_to_noise_ratio(X, Y, mask)
    else:
        R_orig, R_recv, m2 = procrustes(X, Y)
        psnr = peak_signal_to_noise_ratio(R_orig, R_recv, mask)
    # procrustes produces squared error, not the mean error
    ndim = np.prod(X.shape)
    rmse = np.sqrt(m2 / ndim)
    return rmse, psnr


def adjusted_mutual_information_score(X, class_labels):
    X = standardize(X, axis = 0, center = True, scale = False)
    # we know the true clusters
    n_clusters = len(set(class_labels))
    distance_matrix = skmetrics.pairwise.pairwise_distances(X, metric='euclidean')

    # Mutual Information Score using exact n_clusters
    model_exact = AgglomerativeClustering(
        n_clusters = n_clusters, 
        linkage = 'average', 
        metric = 'precomputed')
    class_pred_exact = model_exact.fit_predict(distance_matrix)
    mi_score = skmetrics.mutual_info_score(class_labels, class_pred_exact)

    # Adjusted Mutual Information Score using n_clusters + 2
    model_approx = AgglomerativeClustering(
        n_clusters = n_clusters + 2, 
        linkage = 'average', 
        metric = 'precomputed')
    class_pred_approx = model_approx.fit_predict(distance_matrix)
    adj_mi_score = skmetrics.adjusted_mutual_info_score(class_labels, class_pred_approx)
    return mi_score, adj_mi_score


def coupled_procrustes_per_factor_scaling(L_true, F_true, L_hat, F_hat, dim_policy="zerofill", eps=1e-8):
    """
    Coupled alignment of (L_hat, F_hat) to (L_true, F_true).

    Step 1: find one orthogonal matrix R from F_hat -> F_true:
        min_R || F_true - F_hat R ||_F

    Step 2: after rotation, fit one scale per factor/column:
        d_k = argmin_d || F_true[:,k] - d * (F_hat R)[:,k] ||_2^2

    Then:
        F_aligned = (F_hat R) D
        L_aligned = (L_hat R) D^{-1}

    so that:
        L_aligned @ F_aligned.T == (approximately) L_hat @ F_hat.T
    exactly, except for columns where scale handling hits numerical safeguards.
    """
    def has_no_matrix_signal(X):
        #return (np.linalg.norm(X, ord="fro") > eps) and (np.ptp(X) > eps)
        return np.ptp(X) == 0

    F_true_m, F_hat_m = match_latent_dimensions(F_true, F_hat, dim_policy = dim_policy)
    L_true_m, L_hat_m = match_latent_dimensions(L_true, L_hat, dim_policy = dim_policy)

    # gleanr sometimes produces a single column of zero values
    # We can't rescue with alignment. In fact, procrustes returns error.
    if has_no_matrix_signal(F_hat_m):
        k = F_hat_m.shape[1]
        R = np.eye(k) # no rotation, identity matrix
        scales_safe = np.ones(k, dtype=float)
        F_aligned = F_hat_m.copy()
        L_aligned = L_hat_m.copy()
    else:
        # Shared orthogonal alignment
        R, _ = orthogonal_procrustes(F_hat_m, F_true_m)
    
        F_rot = F_hat_m @ R
        L_rot = L_hat_m @ R
    
        # One scale per aligned column
        numer = np.sum(F_rot * F_true_m, axis=0)
        denom = np.sum(F_rot * F_rot, axis=0)
        scales = np.ones(F_rot.shape[1], dtype=float)
        # valid mask for columns with enough signal
        valid = denom > eps
        scales[valid] = numer[valid] / denom[valid]
        # Clip nearly zero scales for numerical stability. But, keep the sign.
        # Only protect genuinely fitted scales, not padded/invalid columns
        tiny_valid = valid & (np.abs(scales) < eps)
        scales_safe = scales.copy()
        scales_safe[tiny_valid] = np.where(scales_safe[tiny_valid] >= 0, eps, -eps)
    
        F_aligned = F_rot * scales_safe.reshape(1, -1)
        L_aligned = L_rot / scales_safe.reshape(1, -1)

    return {
        "L_true": L_true_m,
        "F_true": F_true_m,
        "L_aligned": L_aligned,
        "F_aligned": F_aligned,
        "rotation": R,
        "scales": scales_safe,
    }

def rebalance_to_unit_F(L, F, eps=1e-12, contrib_tol=1e-12):
    """
    Product-preserving rescaling:
        L F.T = (L * ||F_j||) (F_j / ||F_j||).T

    After this, columns of F have norm 1 when nonzero.
    """
    F_norm = np.sqrt(np.sum(F * F, axis=0))
    L_norm = np.sqrt(np.sum(L * L, axis=0))

    L_new = L.copy()
    F_new = F.copy()

    # Columns that can be safely rescaled.
    rescalable = F_norm > eps

    L_new[:, rescalable] = L_new[:, rescalable] * F_norm[rescalable].reshape(1, -1)
    F_new[:, rescalable] = F_new[:, rescalable] / F_norm[rescalable].reshape(1, -1)

    # Truly zero F columns contribute nothing meaningful.
    L_new[:, ~rescalable] = 0.0
    F_new[:, ~rescalable] = 0.0
    
    # Rebalancing impact: 0 means no change; 1 means typical 10x correction.
    # Contribution from each column
    contrib = L_norm * F_norm
    if np.max(contrib) > eps:
        is_contributing = contrib > contrib_tol * np.max(contrib)
    else:
        is_contributing = contrib > eps
    impact_active = rescalable & is_contributing
    
    if np.any(impact_active):
        log_dev = np.log10(np.maximum(F_norm[impact_active], eps))

        weights = contrib[impact_active]
        weights = weights / np.sum(weights)

        impact_log10 = np.sqrt(np.sum(weights * log_dev**2))
    else:
        impact_log10 = np.nan

    return L_new, F_new, impact_log10
