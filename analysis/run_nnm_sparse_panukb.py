
import os
import numpy as np
import pandas as pd
import pickle
from nnwmf.optimize import FrankWolfe

def class_to_dict(classname, property_list = None):
    model = dict()
    if property_list is None:
        property_list = [ x for x in vars(classname).keys() if x not in ["logger_", "nnm_"] ]
    for info in property_list:
        model[info] = getattr(classname, info)
    return model

dscout_dir = "/gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/panukb"
data_filename = os.path.join(dscout_dir, f"ukbb/ukbb_1.pkl")
with (open(data_filename, "rb")) as fh:
    data = pickle.load(fh)
X_cent = data['Ztrue']
Z_cent = data['Z']
Z_mask = data['Zmask']

rank_seq = [128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0]

for i, r in enumerate(rank_seq):
    model = FrankWolfe(model = 'nnm-sparse', max_iter = 10000, svd_max_iter = 50, show_progress = True)
    model.fit(Z_cent, (r, 1.0), mask = Z_mask)
    nnm_dict = class_to_dict(model)
    out_filename = os.path.join(dscout_dir, f"nnm_sparse/ukbb_1_nnm_sparse_{r}.pkl")
    with open(out_filename, "wb") as fh:
        pickle.dump(nnm_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
