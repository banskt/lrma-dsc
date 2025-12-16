# This file contains simple implementations of
# matrix factorization methods for Z-scores.

fit_flashier <- function(X, k, var_type = c(1,2), L_prior = ebnm_point_normal, F_prior = ebnm_normal, backfit = TRUE) {
    # Most general possible variance structure allowed by flashier
    # will be var_type = c(1,2), 
    # var_type = 1 estimates a precision parameter for each row.
    # var_type = 2 estimates a precision parameter for each column.
    # Default is var_type = 0, estimates a single precision parameter.
    out <- flashier::flash(X, 
                           greedy_Kmax = k, 
                           var_type = var_type, 
                           ebnm_fn = c(L_prior, F_prior),
                           backfit = backfit,
                           verbose = 0)
    return(list(k = out$n_factors, L = out$L_pm, F=out$F_pm, S2=out$pve))
}
