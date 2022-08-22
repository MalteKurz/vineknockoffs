import numpy as np


indep_cop_funs = {'cdf': lambda par, u, v: u * v,
                  'pdf': lambda par, u, v: np.ones_like(u),
                  'll': lambda par, u, v: np.zeros_like(u),
                  'd_ll_d_par': lambda par, u, v: np.zeros_like(u),
                  'd_cdf_d_par': lambda par, u, v: np.zeros_like(u),
                  'hfun': lambda par, u, v: u,
                  'vfun': lambda par, u, v: v,
                  'd_hfun_d_par': lambda par, u, v: np.zeros_like(u),
                  'd_vfun_d_par': lambda par, u, v: np.zeros_like(u),
                  'd_hfun_d_v': lambda par, u, v: np.zeros_like(u),
                  'd_vfun_d_u': lambda par, u, v: np.zeros_like(u),
                  'inv_hfun': lambda par, u, v: u,
                  'inv_vfun': lambda par, u, v: v,
                  }
