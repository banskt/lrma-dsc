# This Python script implements the NNM module.

from fit_softimpute import fit_softimpute

X, model = fit_softimpute(
    Z_with_nan, 
    k_true, 
    kfolds = kfolds, 
    seed = None, 
    cv_max_iter = cv_max_iter, 
    max_iter = max_iter,
    )
