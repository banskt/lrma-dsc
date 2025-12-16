#!/bin/bash

#SBATCH --job-name=lrma_dsc
#SBATCH --partition=cpu
#SBATCH --nodes=1            # minimum number of nodes to be allocated
#SBATCH --ntasks=1           # number of tasks
#SBATCH --cpus-per-task=2    # number of cores on the CPU for the task
#SBATCH --mem=200G
#SBATCH --time=0-02:00:00
#SBATCH --output="%x.out"   # keep appending to the same output file
#SBATCH --error="%x.err"    # keep appending to the same error file

source ~/.bashrc

# Load required modules
module load R/4.4.3
module load FlexiBLAS
module load conda/24.3.0
conda activate py311

# set correct number of threads
export NUMEXPR_MAX_THREADS=2
export NUMEXPR_NUM_THREADS=2

# dsc -c 16 -v 3 lrma.dsc
dsc -c 2 lrma_pilot.dsc
