import numpy as np
from scipy import stats as spstats

def distribute_samples_to_classes(Q, n, shuffle = False):
    '''
    Distribute n samples to Q classes
    '''
    rs = 0.6 * np.random.rand(Q) + 0.2 # random sample from [0.2, 0.8)
    z = np.array(np.round((rs / np.sum(rs)) * n), dtype = int)
    z[-1] = n - np.sum(z[:-1])
    tidx = np.arange(n)
    if shuffle:
        np.random.shuffle(tidx)
    bins = np.zeros(Q + 1, dtype = int)
    bins[1:] = np.cumsum(z)
    idx_groups  = [np.sort(tidx[bins[i]:bins[i+1]]) for i in range(Q)]
    labels = [i for idx in range(n) for i in range(Q) if idx in idx_groups[i]]
    return idx_groups, labels



def get_blockdiag_matrix(n, rholist, rhobg, idx_groups):
    '''
    Generate a block diagonal matrix of size n x n.
    S_ij = 1, if i = j
         = rholist[q],  if i,j \in idx_groups[q]
         = rhobg, otherwise
    '''
    R = np.ones((n, n)) * rhobg

    for i, (idx, rho) in enumerate(zip(idx_groups, rholist)):
        nblock = idx.shape[0]
        xblock = np.ones((nblock, nblock)) * rho
        R[np.ix_(idx, idx)] = xblock

    R[np.diag_indices_from(R)] = 1.0

    return R



def effect_size(n, p, k, Q, h2, g2,
        aq, a0, nsample, 
        cov_design = 'blockdiag',
        sharing_proportion = 1.0,
        shuffle = False,
        seed = None):
    '''
    Get Y = LF' + M + E  where columns of F are orthonormal,
    and L is a blockdiagonal matrix.
    LF' correspond to the shared component of effect sizes,
    the distinct components are given by M, which is sampled
    from a Laplace distribution.
    The noise in the estimate of the effect sizes is given by E.
    '''
    if seed is not None: np.random.seed(seed)
    if not isinstance(h2, np.ndarray):
        h2 = np.ones(n) * h2
    if not isinstance(g2, np.ndarray):
        g2 = np.ones(n) * g2
    if not isinstance(nsample, np.ndarray):
        nsample = np.ones(n) * nsample
    p_shared = int(p * sharing_proportion)

    C_ixgrp, C = distribute_samples_to_classes(Q, n, shuffle = shuffle)
    ggT  = np.sqrt(np.einsum('i,j->ij', g2, g2))
    if cov_design == 'blockdiag':
        rho  = [aq for _ in range(Q)]
        covL = get_blockdiag_matrix(n, rho, a0, C_ixgrp) * ggT
    else:
        covL = np.eye(n) * ggT
    # normalize L for correct variance.
    L  = np.random.multivariate_normal(np.zeros(n), covL, size = k).T 
    L /= np.sqrt(k)
    # F with orthogonal columns
    F  = spstats.ortho_group.rvs(p_shared)[:, :k]
    # M is a sparse matrix of effect sizes
    scaleM = np.sqrt((h2 - g2) * 0.5 / p)
    M = np.random.laplace(np.zeros(n), scaleM, size = (p, n)).T
    # obtain the true effect sizes
    Y = np.zeros((n, p))
    p_choose = np.sort(np.random.choice(p, p_shared, replace = False))
    Y[:, p_choose] = L @ F.T
    Y += M
    # the observed effect size is normally distributed, with mean Y
    # and variance obtained from the residuals.
    stderr = np.sqrt((1 - np.square(Y)) / nsample.reshape(n, 1))
    Yobs = np.random.normal(Y, stderr)
    # observed Z-scores
    Z = Yobs / stderr
    return Z, Yobs, Y, L, F, M, C
