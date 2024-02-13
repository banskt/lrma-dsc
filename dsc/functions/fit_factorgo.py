#!/usr/bin/env python
import os
import tempfile
import shutil
import subprocess
import numpy as np
import pandas as pd

class FactorGoCLI:

    def __init__(self):
        self._process_returncode = 0
        self._process_msg = ""

    @property
    def Wm(self):
        return self._W_mean

    @property
    def Wvar(self):
        return self._W_var

    @property
    def Zm(self):
        return self._Z_mean

    @property
    def Zvar(self):
        return self._Z_var

    @property
    def sorted_index(self):
        return np.array([int(x) for x in self._factor_info[:, 0]])

    @property
    def mean_ard(self):
        return self._factor_info[:, 1]

    @property
    def variance_explained(self):
        return self._factor_info[:, 2]

    @property
    def returncode(self):
        return self._process_returncode

    @property
    def process_msg(self):
        return self._process_msg
        

    def fit(self, X, k, nsample = 10000, numthreads = 1):
        '''
        X is N x P
        N is the number of traits
        P is the number of variants.
        k is the number of latent factors
        nsample is the array of number of samples for each GWAS trait.
        It can also be an integer if all traits have same number of samples.
        '''
        n, p = X.shape
        # initialize to zeros
        self._W_mean = np.zeros((p, k))
        self._W_var  = np.zeros(k)
        self._Z_mean = np.zeros((n, k))
        self._Z_var  = np.zeros((n, k))
        self._factor_info = np.zeros((k, 3))

        # Run the wrapper process
        self.tsv_wrapper(X, k, nsample = nsample, numthreads = numthreads)

    def tsv_wrapper(self, X, k, nsample = 10000, numthreads = 1):
        tmpdir = tempfile.mkdtemp()

        n, p = X.shape
        if not isinstance(nsample, np.ndarray):
            nsample = np.ones(n) * nsample

        # Step 1. Save data in tsv
        zscore_df = pd.DataFrame(X.T)
        zscore_df.columns = [f"z{x+1}" for x in range(n)]
        zscore_df.insert(0, 'rsid', [f"rs{x + 1}" for x in range(p)])
        zscore_fname = os.path.join(tmpdir, "zscore.tsv")
        zscore_df.to_csv(zscore_fname, sep = '\t', index = False)

        # Step 2. Save NSamples in tsv
        N_fname = os.path.join(tmpdir, "sampleN.tsv")
        np.savetxt(N_fname, nsample, fmt='%d', header = 'N')

        # Step 3. Run factorgo
        res_prefix = os.path.join(tmpdir, "simres")
        cmd = ["factorgo"]
        cmd += [zscore_fname, N_fname]
        cmd += ["-k", f"{k}"]
        cmd += ["-o", res_prefix]
        #os.environ['NUMEXPR_NUM_THREADS'] = f"{numthreads}"
        #os.environ['NUMEXPR_MAX_THREADS'] = f"{numthreads}"
        process = subprocess.run(cmd, 
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)

        self._process_returncode = process.returncode
        self._process_msg = process.stderr.decode('utf-8')

        # Step 4. Collect results
        if process.returncode == 0:
            self._W_mean = np.loadtxt(f"{res_prefix}.Wm.tsv.gz")
            self._W_var  = np.loadtxt(f"{res_prefix}.Wvar.tsv.gz")
            self._Z_mean = np.loadtxt(f"{res_prefix}.Zm.tsv.gz")
            self._Z_var  = np.loadtxt(f"{res_prefix}.Zvar.tsv.gz")
            self._factor_info = np.loadtxt(f"{res_prefix}.factor.tsv.gz")

        # Step 5. Delete temp_dir
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
