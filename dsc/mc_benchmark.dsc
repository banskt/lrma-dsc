# DSC for evaluating different low rank matrix approximation methods in different scenarios.
#

DSC:
  python_modules: numpy,
                  clorinn
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/lowrankfit,
                  modules/score
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/mc_benchmark
  replicate:      10
  define:
    mcmethods:    frankwolfe, softimpute
  run:
    benchmark:    input_with_nan * mcmethods * score

# simulate modules
# ===================

input_with_nan: blockdiag_with_nan.py
  n: 200
  p: 2000
  k: 10
  Q: 3
  h2: 0.2
  h2_shared_frac: 0.4
  aq: 0.6
  a0: 0.2
  nsample_minmax: (10000, 40000)
  sharing_proportion: 1.0
  seed: None
  missing_ratio: 0.05, 0.1, 0.2, 0.4
  $Z: Z
  $Z_with_nan: Z_with_nan


# LRMC modules
# ===================
frankwolfe: nnm_mc.py
  Z: $Z
  Z_with_nan: $Z_with_nan
  kfolds: 2
  max_iter: 1000
  cv_max_iter: 1000
  $X: X
  $n_iter: model["n_iter"]
  $time_sec: model["time_sec"]
  $model: model


softimpute: softimpute.py
  Z: $Z
  Z_with_nan: $Z_with_nan
  k_true: 10
  kfolds: 2
  max_iter: 1000
  cv_max_iter: 1000
  $X: X
  $n_iter: model["n_iter"]
  $time_sec: model["time_sec"]
  $model: model


# Analysis modules
# ===================
score: score_mc.py
  Z_true: $Z
  Z_est: $X
  Z_with_nan: $Z_with_nan
  $test_rmse: test_rmse
