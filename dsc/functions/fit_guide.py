import numpy as np
import sys
sys.path.append("/gpfs/commons/home/sbanerjee/software/GUIDE")
import GUIDE

def guide_factors(X, k):
    X_cent = X - np.mean(X, axis = 0, keepdims = True)
    F, Lt, Sc, mix = GUIDE.guide(X_cent.T, L=k, mean_center = True, standardize = False)
    L = Lt.T
    S2 = np.square(Sc)
    return L, F, S2
