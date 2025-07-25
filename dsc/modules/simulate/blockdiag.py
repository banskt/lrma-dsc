#
import simulate
import numpy as np

g2 = h2 * h2_shared_frac
nsample = np.random.uniform(nsample_minmax[0], nsample_minmax[1], n)
Z, effect_size_obs, effect_size_true, L, F, M, C = \
    simulate.effect_size(
        n, p, k, Q, h2, g2, aq, a0, nsample,
        sharing_proportion = sharing_proportion,
        cov_design = 'blockdiag', shuffle = False,
        seed = seed)
