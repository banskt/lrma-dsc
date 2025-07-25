# This Python script implements the RPCA cross-validation module.

from fit_lowrank import rpca, rpca_lambda_cv
import numpy as np

cvlmb, cvrmse = rpca_lambda_cv(Z, Zmask, Ztrue, max_iter = max_iter)
_lmbopt = cvlmb[np.argmin(cvrmse)]
X, M, model = rpca(Z, mask = Zmask, max_iter = max_iter, lmb = _lmbopt)
