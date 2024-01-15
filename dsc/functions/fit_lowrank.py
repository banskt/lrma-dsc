# Functions for low rank approximation used in the simulations

from nnwmf.optimize import IALM
import numpy as np

# This is a placeholder until I find a proper
# dict converter of the class.
# [(p, type(getattr(classname, p))) for p in dir(classname)]
# shows the types, but how to extract the @property methods?
def class_to_dict(classname, property_list = None):
    model = dict()
    if property_list is None:
        property_list = [ x for x in vars(classname).keys() if x not in ["logger_"] ]
    for info in property_list:
        model[info] = getattr(classname, info)
    return model


def do_center_nan(X):
    '''
    X may contain NaN values
    '''
    X_mask = np.isnan(X)
    X_cent = X - np.nanmean(X, axis = 0, keepdims = True)
    X_cent = np.nan_to_num(X_cent, copy = True, nan = 0.0)
    return X_cent, X_mask


def rpca(Y, max_iter = 1000):
    Y_cent, Y_mask  = do_center_nan(Y)
    rpca = IALM(max_iter = 1000, mu_update_method='admm', show_progress = False)
    rpca.fit(Y_cent, mask = Y_mask)
    rpca_dict = class_to_dict(rpca)
    return rpca.L_, rpca.E_, rpca_dict
