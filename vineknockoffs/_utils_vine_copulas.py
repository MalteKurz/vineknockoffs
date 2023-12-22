import numpy as np
from scipy.stats import kendalltau
from python_tsp.exact import solve_tsp_dynamic_programming

try:
    from rpy2 import robjects
except ImportError:
    _has_rpy2 = False
else:
    _has_rpy2 = True

if _has_rpy2:
    _has_r_tsp = robjects.r('library("TSP", quietly=TRUE, logical.return=TRUE)')[0]
    r_solve_tsp = robjects.r('''
            solve_tsp <- function(one_m_tau_mat) {
                hamilton = TSP::insert_dummy(TSP::TSP(one_m_tau_mat), label = "cut")
                sol = TSP::solve_TSP(hamilton, method = "repetitive_nn")
                order = TSP::cut_tour(sol, "cut")
              return(order)
            }
            ''')
else:
    _has_r_tsp = False


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


def kendall_tau_mat(x):
    n_vars = x.shape[1]
    tau_mat = np.full((n_vars, n_vars), np.nan)
    np.fill_diagonal(tau_mat, 1.)
    for i_var in np.arange(0, n_vars):
        for j_var in np.arange(i_var+1, n_vars):
            tau_mat[i_var, j_var] = kendalltau(x[:, i_var], x[:, j_var])[0]
            tau_mat[j_var, i_var] = tau_mat[i_var, j_var]
    return tau_mat


def d_vine_structure_select(u, tsp_method='r_tsp'):
    tau_mat = 1. - np.abs(kendall_tau_mat(u))

    if tsp_method == 'r_tsp':
        if not (_has_rpy2 and _has_r_tsp):
            raise ImportError('To determine the D-vine structure with method r_tsp the python package rpy2 and the R '
                              'package TSP are required.')
        permutation = r_solve_tsp(tau_mat)-1
    else:
        assert tsp_method == 'py_tsp'
        permutation, _ = solve_tsp_dynamic_programming(tau_mat)
    return permutation
