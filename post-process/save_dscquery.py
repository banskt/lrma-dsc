import dscrutils2py as dscrutils
import os
import pandas as pd
import pickle
import argparse
import sys

def parse_args():

    parser = argparse.ArgumentParser(description='Save dscout to pickle format.')

    parser.add_argument('--out',
                        type=str,
                        dest='outfile',
                        metavar='FILE',
                        required=True,
                        help='Name of output file')

    parser.add_argument('--dsc',
                        type=str,
                        dest='dscdir',
                        metavar='FILE',
                        required=True,
                        help='Name of DSC output directory')

    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return options

args = parse_args()

dscdir  = os.path.normpath(args.dscdir)
outfile = os.path.normpath(args.outfile)

targets = ["simulate"] + \
            [f"simulate.{x}" for x in ["n", "p", "k", "h2", "h2_shared_frac", "aq", "nsample_minmax"]] + \
            ["lowrankfit", "mfmethods"] + \
            [f"score.{x}" for x in ["L_rmse", "F_rmse", "Z_rmse", "L_psnr", "F_psnr", "Z_psnr", "MI", "adj_MI", "MI_aligned", "adj_MI_aligned"]]



if os.path.isdir(os.path.dirname(outfile)):
    #dscout = dscrutils.dscquery(os.path.realpath(dscdir), targets, groups = groups)
    dscout = dscrutils.dscquery(os.path.realpath(dscdir), targets)
    dscout.to_pickle(outfile)
else:
    print ("No such file or directory: {:s}".format(os.path.dirname(outfile)))

## one lines for copy-paste
## dsc_name = "lrma_truncate"
## dscdir = f"/gpfs/commons/home/sbanerjee/simdata/low_rank_matrix_approximation_numerical_experiments/{dsc_name}"
## outfile = f"/gpfs/commons/home/sbanerjee/work/npd/lrma-dsc/dsc/results/{dsc_name}_dscout.pkl"
## targets = ["simulate", "simulate.n", "simulate.p", "simulate.k", "simulate.h2", "simulate.h2_shared_frac", "simulate.aq", "lowrankfit", "mfmethods", "score.L_rmse", "score.F_rmse", "score.Z_rmse", "score.L_psnr", "score.F_psnr", "score.Z_psnr", "score.MI", "score.adj_MI", "score.MI_aligned", "score.adj_MI_aligned"]
## dscrutils.dscquery(dscdir, targets).to_pickle(outfile)
