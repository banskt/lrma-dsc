# This Python script implements the FactorGO module.

import numpy as np
from comparison_metrics import matrix_dissimilarity_scores, adjusted_mutual_information_score, mean_squared_error, peak_signal_to_noise_ratio

L_rmse, L_psnr = matrix_dissimilarity_scores(Ltrue, L)
F_rmse, F_psnr = matrix_dissimilarity_scores(Ftrue, F)
Ztrue = Ltrue @ Ftrue.T
Zrecv = L @ F.T
Ztrue = Ztrue - np.mean(Ztrue, axis = 0, keepdims = True)
Zrecv = Zrecv - np.mean(Zrecv, axis = 0, keepdims = True)
Z_rmse = np.sqrt(mean_squared_error(Ztrue, Zrecv))
Z_psnr = peak_signal_to_noise_ratio(Ztrue, Zrecv)
adj_MI = adjusted_mutual_information_score(L, labels)
