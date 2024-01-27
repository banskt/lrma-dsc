# This Python script implements the FactorGO module.

from fit_truncated_svd import truncated_svd

L, F, S2 = truncated_svd(X, k)
Lvar = np.zeros((X.shape[0], k))
Fvar = np.zeros((X.shape[1], k))
