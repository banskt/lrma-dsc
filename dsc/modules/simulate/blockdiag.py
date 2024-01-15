#
import simulate
import numpy as np

g2 = h2 * h2_shared_frac
Z, L, F, M, C = \
    simulate.effect_size(
        n, p, k, Q, h2, g2, aq, a0, nsample,
        cov_design = 'blockdiag', shuffle = False,
        seed=None)
