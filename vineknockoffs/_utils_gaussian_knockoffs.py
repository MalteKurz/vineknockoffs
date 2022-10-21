import numpy as np
import cvxpy as cp
from scipy.linalg import eigh


def sdp_solver(corr_mat, tol_psd=1e-4, tol_s=1e-4):
    n_vars = corr_mat.shape[0]
    tol_mat = np.diag(np.repeat(tol_psd, n_vars))
    s = cp.Variable(n_vars)
    obj_fun = cp.Maximize(cp.sum(s))
    constraints = [cp.diag(s) + tol_mat << 2 * corr_mat]
    constraints += [tol_s <= s, s <= 1.-tol_s]
    print(corr_mat)

    prob = cp.Problem(obj_fun, constraints)
    prob.solve(solver='CVXOPT', verbose=True)

    assert prob.status == cp.OPTIMAL
    return s.value


def ecorr_solver(corr_mat, tol=1e-4):
    n_vars = corr_mat.shape[0]
    lambda_min = eigh(corr_mat, eigvals_only=True, subset_by_index=[0, 0])
    s = np.repeat(np.minimum(2*(lambda_min[0]-tol), 1.), n_vars)
    return s
