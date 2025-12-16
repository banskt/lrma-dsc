# DSC for evaluating different low rank matrix approximation methods in different scenarios.
#

DSC:
  python_modules: numpy,
                  nnwmf
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/lowrankfit,
                  modules/matfactor,
                  modules/score
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/blockdiag
  replicate:      10
  define:
    simulate:     blockdiag, blockdiag_p, blockdiag_k, blockdiag_h2, blockdiag_h2shared, blockdiag_aq
    lowrankfit:   rpca, nnm, nnm_sparse, identical
  run:
    lrma:         simulate * lowrankfit * truncated_svd * score
    factorgo:     simulate * identical * factorgo * score

# simulate modules
# ===================

blockdiag: blockdiag.py
  n: 200
  p: 2000
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
  $effect_size_obs: effect_size_obs
  $effect_size_true: effect_size_true
  $Ltrue: L
  $Ftrue: F
  $Mtrue: M
  $Ctrue: C
  $nsample: nsample

blockdiag_p(blockdiag):
  p: 500, 1000, 5000, 10000

blockdiag_k(blockdiag):
  k: 2, 5, 15, 20

blockdiag_h2(blockdiag):
  h2: 0.05, 0.1, 0.3, 0.4

blockdiag_h2shared(blockdiag):
  h2_shared_frac: 0.2, 0.8, 1.0

blockdiag_aq(blockdiag):
  aq: 0.4, 0.8

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

nnm_sparse: nnm_sparse.py
  Z: $Z
  max_iter: 10000
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
