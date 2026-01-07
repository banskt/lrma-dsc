#!/usr/bin/env bash

# run this command as:
#     source dscquery_cmd.sh [DSC_NAME]

# Set the environment
source ~/.bashrc

# Load required modules
module load R/4.4.3
module load FlexiBLAS
module load conda/24.3.0
conda activate py311

RESDIR="/gpfs/commons/home/sbanerjee/work/npd/lrma-dsc/dsc/results"
SIMDIR="/gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments"
DSCNAME="${1}"

python save_dscquery.py --dsc "${SIMDIR}/${DSCNAME}" --out "${RESDIR}/${DSCNAME}_dscout.pkl"
