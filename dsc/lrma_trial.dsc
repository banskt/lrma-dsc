# A DSC for evaluating prediction accuracy of multiple linear regression
# methods in different scenarios.
#

DSC:
  python_modules: numpy,
                  nnwmf
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/lowrankfit,
                  modules/matfactor,
                  modules/score
  output:         /gpfs/commons/home/sbanerjee/work/npd/lrma-dsc/dsc/trial
  replicate:      1
  define:
    simulate:     blockdiag
    lowrankfit:   rpca, nnm
#    simulate:     blockdiag, blockdiag_p, blockdiag_k, blockdiag_h2, blockdiag_h2shared, blockdiag_aq
#    lowrankfit:   rpca, nnm, nnm_sparse, tsvd, identical
#    matfactor:    svd, factorgo 
  run:
    lrma_trial:   simulate * lowrankfit

# simulate modules
# ===================

blockdiag: blockdiag.py
  n: 500
  p: 1000
  k: 100
  Q: 3
  h2: 0.6
  h2_shared_frac: 0.6
  aq: 0.6
  a0: 0.2
  nsample: 10000
  $Z: Z
  $Ltrue: L
  $Ftrue: F
  $Mtrue: M
  $Ctrue: C

blockdiag_p(blockdiag):
  p: 500, 2000

blockdiag_k(blockdiag):
  k: 10, 50, 200

blockdiag_h2(blockdiag):
  h2: 0.4, 0.8

blockdiag_h2shared(blockdiag):
  h2_shared_frac: 0.2, 0.4, 0.8, 1.0

blockdiag_aq(blockdiag):
  aq: 0.2, 0.4, 0.8

# LRMA modules
# ===================
rpca: rpca.py
  Z: $Z
  max_iter: 10000
  $X: X
  $M: M
  $model: model

nnm: nnm.py
  Z: $Z
  max_iter: 10000
  $X: X
  $model: model

# nnm_sparse: nnm_sparse.py
#   Y: $Y
#   max_iter: 10000

# Factorization modules
# ===================
#svd: svd.py
#  X: $X
#  $L: L
#  $F: F
#  $model: model


# Analysis modules
# ===================

# L_error

# adjusted_MI

# matrix_rank


# Plot modules
# ===================

# Manuscript modules
# ===================
