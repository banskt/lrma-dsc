#!/usr/bin/env bash

dsc-query \
    --target "simulate" "simulate.n" "simulate.p" "simulate.k" "simulate.h2" "simulate.h2_shared_frac" "simulate.aq" "lowrankfit" "matfactor" "score.L_rmse" "score.F_rmse" "score.Z_rmse" "score.L_psnr" "score.F_psnr" "score.Z_psnr" "score.adj_MI" \
    --groups "matfactor: truncated_svd, factorgo" \
    --verbosity 3 \
    --output "${2}" "${1}"
