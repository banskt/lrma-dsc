# This Python script implements the NNM module.

from fit_lowrank import nnm_sparse

X, M, model = nnm_sparse(Z, max_iter = max_iter)
