# This Python script implements the GUIDE module from Lazarev et. al.

from fit_guide import guide_factors

L, F, S2 = guide_factors(X, k)
Lvar = np.zeros((X.shape[0], k))
Fvar = np.zeros((X.shape[1], k))
