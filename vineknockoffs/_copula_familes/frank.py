import numpy as np

# from ._utils_copula_families_sympy import sym_copula_derivs_one_par, write_sympy_expr, copula_derivs_one_par
# from sympy import diff, log, exp, sqrt
# u_sym, v_sym, par_sym = symbols('u v par')

# frank_cdf_sym = - 1/par_sym * \
#                 log(1/(1 - exp(-par_sym)) *
#                     (1 - exp(-par_sym) - (1 - exp(-par_sym*u_sym)) * (1 - exp(-par_sym*v_sym))))
# frank_cop_funs = copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, par_sym)
# frank_sym_dict = sym_copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(frank_sym_dict, './vineknockoffs/sym_copula_expr/frank.csv')


def _cdf(par, u, v):
    # obtained with sympy
    res = -np.log((np.expm1(-par) + np.expm1(-par*u)*np.expm1(-par*v))/np.expm1(-par))/par
    return res


def _hfun(par, u, v):
    # obtained with sympy
    res = -np.exp(-par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
    return res


def _vfun(par, u, v):
    # obtained with sympy
    res = -np.exp(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
    return res


def _pdf(par, u, v):
    # obtained with sympy
    res = par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)\
          / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
    return res


def _ll(par, u, v):
    # obtained with sympy
    res = np.log(-par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par)
                 / (np.expm1(-par) + np.expm1(-par*u)*np.expm1(-par*v))**2)
    return res


def _d_ll_d_par(par, u, v):
    # obtained with sympy
    res = (-par*u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
           - par*u*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
           - par*u*np.exp(-2*par*u)*np.exp(-par*v)*np.expm1(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
           - par*v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
           - par*v*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
           - par*v*np.exp(-par*u)*np.exp(-2*par*v)*np.expm1(-par*u)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
           + par*(-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u)
                  - np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
           + par*(-2*u*np.exp(-par*u)*np.expm1(-par*v) - 2*v*np.exp(-par*v)*np.expm1(-par*u)
                  - 2*np.exp(-par))*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**3 + np.exp(-par*u)*np.exp(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
           + np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)
           / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)\
          / (par*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))
             + par*np.exp(-par*u)*np.exp(-par*v)*np.expm1(-par*u)*np.expm1(-par*v)
             / (-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2)
    return res


def _d_cdf_d_par(par, u, v):
    # obtained with sympy
    res = (-(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))*np.exp(-par)/np.expm1(-par)**2
           - (u*np.exp(-par*u)*np.expm1(-par*v) + v*np.exp(-par*v)*np.expm1(-par*u) + np.exp(-par))
           / np.expm1(-par))*np.expm1(-par)/(par*(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v)))\
          + np.log((np.expm1(-par) + np.expm1(-par*u)*np.expm1(-par*v))/np.expm1(-par))/par**2
    return res


def _d_hfun_d_par(par, u, v):
    # obtained with sympy
    res = u*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          + v*np.exp(-par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          - (-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u)
             - np.exp(-par))*np.exp(-par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
    return res


def _d_vfun_d_par(par, u, v):
    # obtained with sympy
    res = u*np.exp(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          + v*np.exp(-par*u)*np.exp(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          - (-u*np.exp(-par*u)*np.expm1(-par*v) - v*np.exp(-par*v)*np.expm1(-par*u)
             - np.exp(-par))*np.exp(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
    return res


def _d_hfun_d_v(par, u, v):
    # obtained with sympy
    res = par*np.exp(-par*v)*np.expm1(-par*u)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          + par*np.exp(-2*par*v)*np.expm1(-par*u)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
    return res


def _d_vfun_d_u(par, u, v):
    # obtained with sympy
    res = par*np.exp(-par*u)*np.expm1(-par*v)/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))\
          + par*np.exp(-2*par*u)*np.expm1(-par*v)**2/(-np.expm1(-par) - np.expm1(-par*u)*np.expm1(-par*v))**2
    return res


frank_cop_funs = {'cdf': _cdf,
                  'pdf': _pdf,
                  'll': _ll,
                  'd_ll_d_par': _d_ll_d_par,
                  'd_cdf_d_par': _d_cdf_d_par,
                  'hfun': _hfun,
                  'vfun': _vfun,
                  'd_hfun_d_par': _d_hfun_d_par,
                  'd_vfun_d_par': _d_vfun_d_par,
                  'd_hfun_d_v': _d_hfun_d_v,
                  'd_vfun_d_u': _d_vfun_d_u
                  }
