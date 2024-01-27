#!/usr/bin/env bash

# run this command as:
#     source dscquery_cmd.sh [DSC_NAME]

RESDIR="/gpfs/commons/home/sbanerjee/work/npd/lrma-dsc/dsc/results"
SIMDIR="/gpfs/commons/groups/knowles_lab/sbanerjee/low_rank_matrix_approximation_numerical_experiments"
DSCNAME="${1}"

python save_dscquery.py --dsc "${SIMDIR}/${DSCNAME}" --out "${RESDIR}/${DSCNAME}_dscout.pkl"
