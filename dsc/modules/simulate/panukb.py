#
import os
import numpy as np
import pandas as pd
import simulate

#zscore_df = pd.read_pickle(os.path.join(data_dir, f"modselect/zscore_h2{h2_cut}_pval{pval_cut}.pkl"))
#trait_df  = pd.read_pickle(os.path.join(data_dir, f"modselect/traits_h2{h2_cut}.pkl"))

zscore_df = pd.read_pickle(os.path.join(data_dir, f"modselect/zscore_noRx.pkl"))

X = np.array(zscore_df.values.T)
X_cent = X - np.mean(X, axis = 0, keepdims = True)

Z_mask = simulate.generate_mask(X.shape[0], X.shape[1], mask_ratio)
Z_cent = simulate.generate_masked_input(X_cent, Z_mask)
