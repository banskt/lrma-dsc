
import numpy as np
import time
from soft_impute import SoftImpute
from collections import defaultdict
from clorinn.utils import model_errors as merr

def generate_fold_labels(Y, kfolds=2, test_size=None, do_shuffle=True, nan_label=0):
    """
    Provides train/test fold labels for entries of Y, ignoring NaNs.
    Reproduced from Clorinn module.

    Parameters
    ----------
    Y : array (n x p)
        Input matrix possibly containing NaNs.
    kfolds : int
        Number of cross-validation folds.
    test_size : float in (0,1) or None
        Proportion of entries in each fold. If None, uniform split.
    do_shuffle : bool
        Whether to shuffle observed entries before assigning folds.
    nan_label : int
        Label used for NaN locations (default 0)
    """

    n, p = Y.shape
    mask = ~np.isnan(Y)              # True for observed entries
    idx = np.where(mask.ravel())[0]  # Indices of observed entries (flattened)

    n_obs = len(idx)
    fold_labels = np.full(n * p, nan_label, dtype=int)  # All NaN by default

    # Number of observed entries per fold
    if test_size is None:
        ntest = n_obs // kfolds
    else:
        ntest = int(test_size * n_obs)

    # Shuffle observed indices if requested
    if do_shuffle:
        np.random.shuffle(idx)

    # Assign fold labels only to observed entries
    for k in range(kfolds):
        start = k * ntest
        end = (k + 1) * ntest if k < kfolds - 1 else n_obs
        fold_labels[idx[start:end]] = k + 1

    return fold_labels.reshape(n, p)

def generate_masked_input(Y, mask):
    """
    Puts nan value to maked indices of the input matrix Y
    without overwriting.
    """
    Y_miss = Y.copy()
    Y_miss[mask] = np.nan
    return Y_miss

def softimpute_cv(Z_with_nan, k_true, kfolds = 2, seed = None, cv_max_iter = 1000):
    
    n, p = Z_with_nan.shape
    softimpute_J_list = sorted({max(2, k_true // 2), k_true, min(min(n, p), 2 * k_true)})
    softimpute_lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    fold_labels = generate_fold_labels(Z_with_nan, kfolds = kfolds)
    cv_results = list()
    
    for k in range(kfolds):
        cv_mask = fold_labels == k + 1
        Z_cv_test  = generate_masked_input(Z_with_nan, cv_mask)
        Z_cv_train = generate_masked_input(Z_with_nan, ~cv_mask)    
    
        data_scale = float(np.nanstd(Z_cv_train))
        if not np.isfinite(data_scale) or data_scale == 0:
            data_scale = float(np.nanstd(Z_with_nan)) + 1e-8

        for J in softimpute_J_list:
            for lam in softimpute_lambda_list:
                t0 = time.perf_counter()
                model = SoftImpute(
                    J = J,
                    lambda_ = lam * data_scale,
                    thresh = 1e-5,
                    maxit = cv_max_iter,
                    random_state = seed,
                    verbose = False,
                )
                model.fit(Z_cv_train)
                X_hat = model.predict(Z_cv_train)
                t1 = time.perf_counter()
                res = {
                    "fold": k + 1,
                    "J": J,
                    "lambda_" : lam * data_scale,
                    "lambda_multiplier": lam,
                    "rmse_test": merr.get(Z_cv_test, X_hat, method = 'rmse'),
                    "rmse_train": merr.get(Z_cv_train, X_hat, method = 'rmse'),
                    "time_sec": t1 - t0,
                }
                cv_results.append(res)   
    return cv_results

def get_softimpute_cv_optimum(cv_results):
    acc = defaultdict(list)

    for d in cv_results:
        acc[(d["J"], d["lambda_multiplier"])].append(d["rmse_test"])
    
    mean_rmse = {key: np.mean(values) for key, values in acc.items()}
    best_J, best_lambda_multiplier = min(mean_rmse, key=mean_rmse.get)
    return best_J, best_lambda_multiplier, mean_rmse

def fit_softimpute(Z_with_nan, k_true, kfolds = 2, seed = None, cv_max_iter = 1000, max_iter = 1000):
    """
    Fits SoftImpute on matrix with NaNs. Returns X_hat and diagnostics.
    """

    t0 = time.perf_counter()
    si_cv = softimpute_cv(Z_with_nan, k_true, 
        kfolds = kfolds, 
        seed = seed, 
        cv_max_iter = cv_max_iter)

    J_opt, lam_opt, cv_test_rmse = get_softimpute_cv_optimum(si_cv)
    t1 = time.perf_counter()
    
    t2 = time.perf_counter()
    data_scale = np.nanstd(Z_with_nan)
    model = SoftImpute(
        J = J_opt,
        thresh = 1e-5,
        lambda_ = lam_opt * data_scale,
        maxit = max_iter,
        random_state = seed,
        verbose = False,
    )
    model.fit(Z_with_nan)
    X_hat = model.predict(Z_with_nan)
    t3 = time.perf_counter()

    out = {
        "time_sec": t3 - t2,
        "cv_time_sec": t1 - t0,
        "J": J_opt,
        "lambda_": lam_opt * data_scale,
        "ratios": model.ratios,
        "n_iter": len(model.ratios),
    }

    return X_hat, out
