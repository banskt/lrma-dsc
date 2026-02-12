# Functions for low rank approximation used in the simulations

import numpy as np
import time
from clorinn.optimize import IALM
from clorinn.optimize import FrankWolfe, FrankWolfe_CV

# This is a placeholder until I find a proper
# dict converter of the class.
# [(p, type(getattr(classname, p))) for p in dir(classname)]
# shows the types, but how to extract the @property methods?
def class_to_dict(classname, property_list = None):
    model = dict()
    if property_list is None:
        property_list = [ x for x in vars(classname).keys() if x not in ["logger_", "nnm_"] ]
    for info in property_list:
        model[info] = getattr(classname, info)
    return model


def nuclear_norm(X):
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    return np.sum(s)


def get_masked_rmse(original, recovered, mask = None):
    if mask is None: mask = np.ones_like(original)
    n = np.sum(mask)
    mse = np.nansum(np.square((original - recovered) * mask)) / n
    return np.sqrt(mse)


def do_center_nan(X, mask = None):
    '''
    X may contain NaN values.
    If mask is not None, set mask to NaN.
    '''
    X_nan = X.copy()
    if mask is not None:
        X_nan[mask] = np.nan
    X_mask = np.isnan(X_nan)
    X_cent = X_nan - np.nanmean(X_nan, axis = 0, keepdims = True)
    # X_cent = np.nan_to_num(X_cent, copy = True, nan = 0.0)
    return X_cent, X_mask


def rpca_lambda_cv(Y, Ymask, Ytrue, max_iter = 1000, ncvseq = 20):
    lmb_min = 0.5 / np.sqrt(np.max(Y.shape))
    lmb_max = 1.0 / np.sqrt(np.min(Y.shape))
    lmb_seq = np.logspace(np.log10(lmb_min), np.log10(lmb_max), ncvseq)
    rmse_seq = [0. for i in range(ncvseq)]
    Y_nan_zero = np.nan_to_num(Y, copy = True, nan = 0.0)
    for i in range(ncvseq):
        model = IALM(max_iter = max_iter, mu_update_method='admm', show_progress = True, print_skip = 100)
        model.fit(Y_nan_zero, mask = Ymask, lmb = lmb_seq[i])
        rmse_seq[i] = get_masked_rmse(Ytrue, model.L_, mask = Ymask)
    return lmb_seq, rmse_seq


def rpca(Y, max_iter = 1000, mask = None, lmb = None):
    Y_cent_nan, Y_mask  = do_center_nan(Y, mask = mask)
    # RPCA cannot handle NaN values.
    Y_cent = np.nan_to_num(Y_cent_nan, copy = True, nan = 0.0)
    rpca = IALM(max_iter = 1000, mu_update_method='admm', show_progress = False)
    rpca.fit(Y_cent, mask = Y_mask, lmb = lmb)
    rpca_dict = class_to_dict(rpca)
    return rpca.L_, rpca.E_, rpca_dict


def nnm(Y, max_iter = 1000, cv_max_iter = 1000, mask = None):
    Y_cent, Y_mask  = do_center_nan(Y, mask = mask)
    nnmcv = FrankWolfe_CV(kfolds = 2, model = 'nnm', max_iter = cv_max_iter)
    nnmcv.fit(Y_cent)
    rank = nnmcv._optimized_rank()
    nnm = FrankWolfe(model = 'nnm', max_iter = max_iter, svd_max_iter = 50, show_progress = False, debug = False)
    nnm.fit(Y_cent, rank, mask = mask)
    nnm_dict = class_to_dict(nnm)
    nnm_dict["train_error"] = nnmcv.train_error_
    nnm_dict["test_error"] = nnmcv.test_error_
    return nnm.X, nnm_dict


def nnm_sparse(Y, max_iter = 1000, cv_max_iter = 1000, mask = None):
    Y_cent, Y_mask  = do_center_nan(Y, mask = mask)
    nnmcv = FrankWolfe_CV(kfolds = 2, model = 'nnm-sparse', max_iter = cv_max_iter)
    nnmcv.fit(Y_cent)
    rank = nnmcv._optimized_rank()
    lmb  = 1.0
    nnm = FrankWolfe(model = 'nnm-sparse', max_iter = max_iter, svd_max_iter = 50, show_progress = False, debug = False)
    nnm.fit(Y_cent, (rank, lmb), mask = mask)
    nnm_dict = class_to_dict(nnm)
    nnm_dict["train_error"] = nnmcv.train_error_
    nnm_dict["test_error"] = nnmcv.test_error_
    return nnm.X, nnm.M_, nnm_dict


def fit_frankwolfe(Y_obs_with_nan, kfolds = 2, max_iter = 1000, cv_max_iter = 1000, rank_seq = None):
    """
    Fits clorinn FrankWolfe on matrix with NaNs. Returns X_hat and diagnostics.
    """

    ### Step 1. Cross validation
    t0 = time.perf_counter()
    nnmcv = FrankWolfe_CV(kfolds = kfolds, model = 'nnm', max_iter = cv_max_iter)
    nnmcv.fit(Y_obs_with_nan, rseq = rank_seq)
    rank = nnmcv._optimized_rank()
    t1 = time.perf_counter()

    ### Step 2. Fit with optimum rank
    t2 = time.perf_counter()
    model = FrankWolfe(model = 'nnm', max_iter = max_iter, svd_max_iter = 50, show_progress = False, debug = False)
    model.fit(Y_obs_with_nan, rank)
    t3 = time.perf_counter()

    X_hat = model.X
    out = {
        "time_sec": t3 - t2,
        "time_cv_sec": t1 - t0,
        "n_iter": len(model.steps),
        "rank": rank,
        "final_duality_gap": model.duality_gaps[-1],
        "final_step": model.steps[-1],
        "final_fx": model.fx[-1],
        "cv_test_error": nnmcv.test_error,
        "cv_train_error": nnmcv.training_error,
        "duality_gaps": model.duality_gaps,
        "fx": model.fx,
        "steps": model.steps,
        "model_dict": class_to_dict(model),
    }
    return X_hat, out

