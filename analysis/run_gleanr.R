library(gleanr)
library(data.table)
library(devtools)

beta <- fread(system.file("extdata", "sim1.effect_sizes.txt", package = "gleanr"))
se <- fread(system.file("extdata", "sim1.std_error.txt", package = "gleanr"))
c.mat <- as.matrix(fread(system.file("extdata", "sim1.c_matrix.txt", package = "gleanr")))
c_se.mat <- as.matrix(fread(system.file("extdata", "sim1.c_se_matrix.txt", package = "gleanr")))

trait_names <- names(beta)[-1]
snp_names <- unlist(beta$SNP)

beta_m <- as.matrix(beta[,-1])
W_s <- 1/as.matrix(se[,-1])

res <- gleanr(beta_m,W_s, snp_names, trait_names, C=c.mat, covar_se=c_se.mat, K="GRID",conv_objective=0.005, verbosity=0, save_out=FALSE)

source("https://raw.githubusercontent.com/aomdahl/gleanr_workflow/refs/heads/main/src/plot_functions.R")
plotFactors(res$V,trait_names = trait_names,title = "gleanr tutorial heatmap")
