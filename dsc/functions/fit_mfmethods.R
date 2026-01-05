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

fit_gleanr <- function (Z, effect_sizes, k) {
    ### Inputs ###
    # Z: N traits x P snps, Z-scores
    # effect_sizes: N traits x P snps
    # 
    ### gleanr expects:
    # beta: P snps x N traits
    # std_err: P snps x N traits
    # covar_matrix: NxN matrix of estimated correlation due to sample sharing; this may be estimated using LDSC
    

    beta <- t(as.matrix(effect_sizes))
    Zmat <- t(as.matrix(Z))
    # W <- 1 / as.matrix(std_err)
    W <- Zmat / beta
    # guard against division by 0 / Inf / NaN
    W[!is.finite(W)] <- 0   
    # guard against finite but huge values
    W <- pmin(pmax(W, -1e6), 1e6)

    # Dummy SNP ids (length P)
    snp_ids <- paste0("snp_", seq_len(nrow(beta)))

    # Dummy trait names (length N)
    trait_names <- paste0("trait_", seq_len(ncol(beta)))

    # Create covariance matrix, and check for finite elements
    covar_matrix <- round(stats::cor(Zmat, use = "pairwise.complete.obs"), 4)
    covar_matrix[!is.finite(covar_matrix)] <- 0
    covar_matrix <- (covar_matrix + t(covar_matrix)) / 2
    diag(covar_matrix) <- 1

    res <- gleanr::gleanr(beta, W, snp_ids, trait_names,
                          C=covar_matrix,
                          K=k, 
                          fixed_ubiq=TRUE, 
                          shrinkWL=0.5, 
                          conv_objective=0.005, 
                          verbosity=0, 
                          save_out=FALSE)
    return(list(k = res$K, L = res$V, F=res$U, S2=as.vector(res$PVE)))
}
