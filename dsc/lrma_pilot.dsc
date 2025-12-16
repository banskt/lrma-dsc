# DSC for evaluating different low rank matrix approximation methods in different scenarios.
#

DSC:
  R_libs:         flashier 
  python_modules: numpy,
                  clorinn
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/lowrankfit,
                  modules/matfactor,
                  modules/score
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/lrma_pilot
  replicate:      1
  define:
    simulate:     blockdiag
    lowrankfit:   rpca, nnm, nnm_sparse
    mfmethods:    truncated_svd, factorgo, flashier
  run:
    clorinn:      simulate * lowrankfit * truncated_svd * score
    benchmark:    simulate * identical * mfmethods * score
#    trial:        simulate * identical * flashier * score

# simulate modules
# ===================

blockdiag: blockdiag.py
  n: 200
  p: 500
  k: 10
  Q: 3
  h2: 0.2
  h2_shared_frac: 0.5
  aq: 0.6
  a0: 0.2
  nsample_minmax: (10000, 40000)
  sharing_proportion: 1.0
  seed: None
  $Z: Z
  $Zmask: None
  $effect_size_obs: effect_size_obs
  $effect_size_true: effect_size_true
  $Ltrue: L
  $Ftrue: F
  $Mtrue: M
  $Ctrue: C
  $nsample: nsample

# LRMA modules
# ===================
rpca: rpca.py
  Z: $Z
  Zmask: $Zmask
  max_iter: 10000
  $X: X
  $M: M
  $model: model

nnm: nnm.py
  Z: $Z
  Zmask: $Zmask
  max_iter: 10000
  cv_max_iter: 1000
  $X: X
  $model: model

nnm_sparse: nnm_sparse.py
  Z: $Z
  Zmask: $Zmask
  max_iter: 10000
  cv_max_iter: 1000
  $X: X
  $M: M
  $model: model

identical: identical.py
  Z: $Z
  $X: X

## Factorization modules
## ===================
factorgo: factorgo.py
  X: $X
  nsample: $nsample
  k: 10
  $L_est: L
  $Lvar_est: Lvar
  $F_est: F
  $Fvar_est: Fvar
  $S2: S2
  $ard_post_mean: ard_post_mean

truncated_svd: truncated_svd.py
  X: $X
  k: 10
  $L_est: L
  $F_est: F
  $S2: S2

flashier: flashier.R
  X: $X
  L_prior: ebnm_point_normal
  F_prior: ebnm_normal
  k: 10
  var_type: c(1,2)
  backfit: TRUE
  $L_est: out$L
  $F_est: out$F
  $S2: out$S2


# Analysis modules
# ===================
score: score.py
  Ltrue: $Ltrue
  Ftrue: $Ftrue
  L: $L_est
  F: $F_est
  labels: $Ctrue
  $L_rmse: L_rmse
  $F_rmse: F_rmse
  $Z_rmse: Z_rmse
  $L_psnr: L_psnr
  $F_psnr: F_psnr
  $Z_psnr: Z_psnr
  $adj_MI: adj_MI

# L_error

# adjusted_MI

# matrix_rank
