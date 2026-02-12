# This Python script implements the FactorGO module.

import numpy as np
from comparison_metrics import matrix_dissimilarity_scores

test_mask = np.isnan(Z_with_nan)
test_rmse = root_mean_squared_error(Z_true, Z_est, mask = test_mask)
