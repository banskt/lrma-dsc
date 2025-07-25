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
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/rpca_runtime_izzy
  replicate:      10
  define:
    simulate:     blockdiag
    lowrankfit:   rpca
  run:            simulate * lowrankfit

# simulate modules
# ===================

blockdiag: blockdiag.py
  n: 100
  p: 8000
  k: 10
  Q: 3
  h2: 0.2
  h2_shared_frac: 0.5
  aq: 0.6
  a0: 0.2
  nsample_minmax: (1000, 4000)
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
