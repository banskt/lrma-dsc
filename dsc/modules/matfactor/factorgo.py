# This Python script implements the FactorGO module.

from fit_factorgo import FactorGoCLI

model = FactorGoCLI()
model.fit(X, k, nsample = nsample)
L    = model.Zm
Lvar = model.Zvar
F    = model.Wm
Fvar = model.Wvar
S2   = model.variance_explained
ard_post_mean = model.mean_ard
