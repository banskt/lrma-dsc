library(dscrutils)
outdir <- "/gpfs/commons/home/sbanerjee/simdata/low_rank_matrix_approximation_numerical_experiments/lrma_pilot"
targets <- c("simulate", "simulate.n", "simulate.p", "simulate.k", "simulate.h2", 
              "simulate.h2_shared_frac", "simulate.aq", "simulate.nsample", 
              "lowrankfit", "mfmethods", "score.L_rmse", "score.F_rmse", "score.Z_rmse", 
              "score.L_psnr", "score.F_psnr", "score.Z_psnr", "score.adj_MI")
dscout <- dscquery(dsc.outdir = outdir, targets = targets)
