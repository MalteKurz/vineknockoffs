import numpy as np


def _uv(par, u, v):
    return u * v


def _u(par, u, v):
    return u


def _v(par, u, v):
    return v


def _ones(par, u, v):
    return np.ones_like(u)


def _zeros(par, u, v):
    return np.zeros_like(u)


indep_cop_funs = {'cdf': _uv,
                  'pdf': _ones,
                  'll': _zeros,
                  'd_ll_d_par': _zeros,
                  'd_cdf_d_par': _zeros,
                  'hfun': _u,
                  'vfun': _v,
                  'd_hfun_d_par': _zeros,
                  'd_vfun_d_par': _zeros,
                  'd_hfun_d_v': _zeros,
                  'd_vfun_d_u': _zeros,
                  'inv_hfun': _u,
                  'inv_vfun': _v,
                  }
