#!/bin/bash

#SBATCH --job-name=lrma_dsc
#SBATCH --partition=cpu
#SBATCH --nodes=1            # minimum number of nodes to be allocated
#SBATCH --ntasks=1           # number of tasks
#SBATCH --cpus-per-task=8    # number of cores on the CPU for the task
#SBATCH --mem=30G
#SBATCH --time=0-06:00:00
#SBATCH --output="%x.out"   # keep appending to the same output file
#SBATCH --error="%x.err"    # keep appending to the same error file

source ~/.bashrc

# Load required modules for NYGC cluster
module load Renv/4.4.3-nygc
module load FlexiBLAS
module load conda/24.3.0
conda activate py311

# set correct number of threads
export _N_JOB_CORES=8          # same as the number of cores requested by this job
export NUMEXPR_MAX_THREADS=${_N_JOB_CORES}
export NUMEXPR_NUM_THREADS=${_N_JOB_CORES}

# dsc -c 16 -v 3 lrma.dsc
# dsc -c ${_N_JOB_CORES} -v 3 -o /gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments/lrma_truncate --truncate lrma.dsc
dsc -s existing -c ${_N_JOB_CORES} lrma.dsc
