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
Z = L @ F.T

# Oracle alignment on raw data
aligned_raw = coupled_procrustes_per_factor_scaling(Ltrue, Ftrue, L, F, dim_policy="zerofill")
Lt_raw = aligned_raw["L_true"]
Ft_raw = aligned_raw["F_true"]
La_raw = aligned_raw["L_aligned"]
Fa_raw = aligned_raw["F_aligned"]
scales_raw = aligned_raw["scales"]

# our simulations use orthonormal factors.
# Rebalance L and F enforcing ||F||_2 = 1
L_bal, F_bal, balancing_impact = rebalance_to_unit_F(L, F)

# Oracle alignment after balancing
aligned_bal = coupled_procrustes_per_factor_scaling(Ltrue, Ftrue, L_bal, F_bal, dim_policy="zerofill")
Lt_bal = aligned_bal["L_true"]
Ft_bal = aligned_bal["F_true"]
La_bal = aligned_bal["L_aligned"]
Fa_bal = aligned_bal["F_aligned"]
scales_bal = aligned_bal["scales"]

# some methods introduce attenuation bias
# remove only the attenuation bias while preserving row-wise and column-wise structure.
# At zero norm, we force L and F to zero.
sg, is_zero_norm = global_calibration(Zt, Z)
Za = sg * Z
# Assign global calibration to L by convention
L_cal = sg * L
F_cal = np.zeros_like(F) if is_zero_norm else F.copy()
L_bal_cal = sg * L_bal
F_bal_cal = np.zeros_like(F_bal) if is_zero_norm else F_bal.copy()

# Oracle alignment after global_calibration
aligned_cal = coupled_procrustes_per_factor_scaling(Ltrue, Ftrue, L_cal, F_cal, dim_policy="zerofill")
Lt_cal = aligned_cal["L_true"]
Ft_cal = aligned_cal["F_true"]
La_cal = aligned_cal["L_aligned"]
Fa_cal = aligned_cal["F_aligned"]
scales_cal = aligned_cal["scales"]

# Oracle alignment after balancing + global calibration
aligned_bal_cal = coupled_procrustes_per_factor_scaling(Ltrue, Ftrue, L_bal_cal, F_bal_cal, dim_policy="zerofill")
Lt = aligned_bal_cal["L_true"]
Ft = aligned_bal_cal["F_true"]
La = aligned_bal_cal["L_aligned"]
Fa = aligned_bal_cal["F_aligned"]
scales = aligned_bal_cal["scales"]

# RMSE
# align
L_rmse_raw = root_mean_squared_error(Lt_raw, La_raw)
F_rmse_raw = root_mean_squared_error(Ft_raw, Fa_raw)
Z_rmse_raw = root_mean_squared_error(Zt, Z)
# balance + align
L_rmse_bal = root_mean_squared_error(Lt_bal, La_bal)
F_rmse_bal = root_mean_squared_error(Ft_bal, Fa_bal)
Z_rmse_bal = root_mean_squared_error(Zt, Z) # balancing does not change Z.
# calibrate + align
L_rmse_cal = root_mean_squared_error(Lt_cal, La_cal)
F_rmse_cal = root_mean_squared_error(Ft_cal, Fa_cal)
Z_rmse_cal = root_mean_squared_error(Zt, Za)
# balance + calibrate + align
L_rmse = root_mean_squared_error(Lt, La)
F_rmse = root_mean_squared_error(Ft, Fa)
Z_rmse = root_mean_squared_error(Zt, Za)

# Relative RMSE / relative Frobenius error
# align
L_rel_rmse_raw = relative_rmse(L_rmse_raw, Lt)
F_rel_rmse_raw = relative_rmse(F_rmse_raw, Ft)
Z_rel_rmse_raw = relative_rmse(Z_rmse_raw, Zt)
# balance + align
L_rel_rmse_bal = relative_rmse(L_rmse_bal, Lt)
F_rel_rmse_bal = relative_rmse(F_rmse_bal, Ft)
Z_rel_rmse_bal = relative_rmse(Z_rmse_bal, Zt)
# calibrate + align
L_rel_rmse_cal = relative_rmse(L_rmse_cal, Lt)
F_rel_rmse_cal = relative_rmse(F_rmse_cal, Ft)
Z_rel_rmse_cal = relative_rmse(Z_rmse_cal, Zt)
# balance + calibrate + align
L_rel_rmse = relative_rmse(L_rmse, Lt)
F_rel_rmse = relative_rmse(F_rmse, Ft)
Z_rel_rmse = relative_rmse(Z_rmse, Zt)

# Optional PSNR; not primary
L_psnr = peak_signal_to_noise_ratio(Lt, La)
F_psnr = peak_signal_to_noise_ratio(Ft, Fa)
Z_psnr = peak_signal_to_noise_ratio(Zt, Za)

# Clustering metrics
MI_raw, adj_MI_raw = adjusted_mutual_information_score(L, labels)
MI_bal, adj_MI_bal = adjusted_mutual_information_score(L_bal, labels)
MI_cal, adj_MI_cal = adjusted_mutual_information_score(L_cal, labels)
MI_bal_cal, adj_MI_bal_cal = adjusted_mutual_information_score(L_bal_cal, labels)
MI_oracle_aligned, adj_MI_oracle_aligned = adjusted_mutual_information_score(La, labels)
