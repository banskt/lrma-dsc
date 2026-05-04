# This Python script implements the scoring of L, F and LF'
# Inputs from pipeline:
#   Ltrue
#   Ftrue
#   L
#   F
#   labels


import numpy as np
from comparison_metrics import (
    global_calibration,
    rebalance_to_unit_F,
    coupled_procrustes_per_factor_scaling,
    root_mean_squared_error,
    relative_rmse,
    peak_signal_to_noise_ratio,
    adjusted_mutual_information_score
)

# True signal
Zt = Ltrue @ Ftrue.T

# Raw estimated signal
Za_raw = L @ F.T

# some methods introduce attenuation bias
# remove only the attenuation bias while preserving row-wise and column-wise structure.
# At zero norm, we force L and F to zero.
sg, is_zero_norm = global_calibration(Zt, Za_raw)
Za = sg * Za_raw
# Assign global calibration to L by convention
L_cal = sg * L
F_cal = np.zeros_like(F) if is_zero_norm else F.copy()

# our simulations use orthonormal factors.
# Rebalance L and F enforcing ||F||_2 = 1
L_bal, F_bal, balancing_impact = rebalance_to_unit_F(L_cal, F_cal)

# Oracle alignment for factor recovery metrics
aligned = coupled_procrustes_per_factor_scaling(Ltrue, Ftrue, L_bal, F_bal, dim_policy="zerofill")
Lt = aligned["L_true"]
Ft = aligned["F_true"]
La = aligned["L_aligned"]
Fa = aligned["F_aligned"]
scales = aligned["scales"]

# RMSE
L_rmse = root_mean_squared_error(Lt, La)
F_rmse = root_mean_squared_error(Ft, Fa)
Z_rmse = root_mean_squared_error(Zt, Za)

# Relative RMSE / relative Frobenius error
L_rel_rmse = relative_rmse(L_rmse, Lt)
F_rel_rmse = relative_rmse(F_rmse, Ft)
Z_rel_rmse = relative_rmse(Z_rmse, Zt)

# Optional PSNR; not primary
L_psnr = peak_signal_to_noise_ratio(Lt, La)
F_psnr = peak_signal_to_noise_ratio(Ft, Fa)
Z_psnr = peak_signal_to_noise_ratio(Zt, Za)

# Clustering metrics
MI_raw, adj_MI_raw = adjusted_mutual_information_score(L, labels)
MI_cal, adj_MI_cal = adjusted_mutual_information_score(L_cal, labels)
MI_oracle_aligned, adj_MI_oracle_aligned = adjusted_mutual_information_score(La, labels)
