# This Python script implements the scoring of L, F and LF'
# Inputs from pipeline:
#   Ltrue
#   Ftrue
#   L
#   F
#   labels


import numpy as np
from comparison_metrics import (
    standardize,
    coupled_procrustes_per_factor_scaling,
    root_mean_squared_error,
    peak_signal_to_noise_ratio,
    adjusted_mutual_information_score
)

# Align L and F jointly to their true values
# so that LF' = Z remains true even after alignment.
aligned = coupled_procrustes_per_factor_scaling(Ltrue, Ftrue, L, F, dim_policy="zerofill")
Lt = aligned["L_true"]
Ft = aligned["F_true"]
La = aligned["L_aligned"]
Fa = aligned["F_aligned"]

# Z need not be aligned, 
# but some methods standardize the input before factorizing.
Zt = standardize(Ltrue @ Ftrue.T, axis = 0, center = True, scale = True)
Za = standardize(L @ F.T, axis = 0, center = True, scale = True)

# Calculate the scores
L_rmse = root_mean_squared_error(Lt, La)
F_rmse = root_mean_squared_error(Ft, Fa)
Z_rmse = root_mean_squared_error(Zt, Za)
L_psnr = peak_signal_to_noise_ratio(Lt, La)
F_psnr = peak_signal_to_noise_ratio(Ft, Fa)
Z_psnr = peak_signal_to_noise_ratio(Zt, Za)

# cluster on aligned loadings or raw loadings?
# La was obtained by aligning to truth, 
# so it uses oracle information.
# But, let's save both.
MI, adj_MI = adjusted_mutual_information_score(L, labels)
MI_aligned, adj_MI_aligned = adjusted_mutual_information_score(La, labels)
