# This Python script implements the NNM module.

from fit_lowrank import fit_frankwolfe, nuclear_norm

fw_rseq = nuclear_norm(Z) * np.array([1e-3, 1e-2, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8])
X, model = fit_frankwolfe(Z_with_nan, kfolds = kfolds, max_iter = max_iter, cv_max_iter = cv_max_iter, rank_seq = fw_rseq)
