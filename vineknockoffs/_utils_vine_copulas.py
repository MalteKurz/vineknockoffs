import numpy as np


def dvine_pcorr(corr_mat):
    dim = corr_mat.shape[0]
    pcorrs = [[np.nan] * j for j in np.arange(dim - 1, 0, -1)]
    for j in np.arange(1, dim):
        tree = j
        for i in np.arange(1, dim-j+1):
            cop = i
            p = np.linalg.inv(corr_mat[i-1:i+j, i-1:i+j])
            pcorrs[tree-1][cop-1] = - p[0, tree] / np.sqrt(p[0, 0] * p[tree, tree])
    return pcorrs
