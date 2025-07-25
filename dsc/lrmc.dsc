# DSC for evaluating different low rank matrix completion using different methods.
#

DSC:
  python_modules: numpy,
                  nnwmf
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/lowrankfit,
                  modules/matfactor,
                  modules/score
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/panukb
  replicate:      1
  define:
    simulate:     ukbb
    lowrankfit:   rpca, nnm, nnm_sparse 
  run:
    lrmc:         simulate * lowrankfit

# simulate modules
# We use the PanUKB data as input.
# We mask 20% of the input data. 
# The goal is to complete the matrix.
# For each method, we use CV to find the optimum threshold of nuclear norm.
# We do not want the CV to know the masked values. 
# After training, we will compare the accuracy of predicting the masked values.
# ===================

ukbb: panukb.py
  data_dir: "/gpfs/commons/home/sbanerjee/work/npd/PanUKB/data"
  h2_cut: "0.1"
  pval_cut: "5e-08"
  mask_ratio: 0.2
  $Ztrue: X_cent
  $Z: Z_cent
  $Zmask: Z_mask


# LRMC modules
# ===================
rpca: rpca_cv.py
  Ztrue: $Ztrue
  Z: $Z
  Zmask: $Zmask
  max_iter: 10000
  $cvlmb: cvlmb
  $cvrmse: cvrmse
  $X: X
  $M: M
  $model: model

nnm: nnm.py
  Z: $Z
  Zmask: $Zmask
  cv_max_iter: 1000
  max_iter: 10000
  $X: X
  $model: model

nnm_sparse: nnm_sparse.py
  Z: $Z
  Zmask: $Zmask
  cv_max_iter: 1000
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


# Plot modules
# ===================

# Manuscript modules
# ===================
