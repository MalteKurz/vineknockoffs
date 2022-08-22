import numpy as np

# from ._utils_copula_families_sympy import sym_copula_derivs_one_par, write_sympy_expr, copula_derivs_one_par
# from sympy import diff, log, exp, sqrt
# u_sym, v_sym, par_sym = symbols('u v par')

# gumbel_cdf_sym = exp(-((-log(u_sym))**par_sym + (-log(v_sym))**par_sym)**(1/par_sym))
# gumbel_cop_funs = copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, par_sym)
# gumbel_sym_dict = sym_copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(gumbel_sym_dict, './gumbel.csv')


def _cdf(par, u, v):
    # obtained with sympy
    res = np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))
    return res


def _hfun(par, u, v):
    # obtained with sympy
    res = -(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (v*((-np.log(u))**par + (-np.log(v))**par)*np.log(v))
    return res


def _vfun(par, u, v):
    # obtained with sympy
    res = -(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (u*((-np.log(u))**par + (-np.log(v))**par)*np.log(u))
    return res


def _pdf(par, u, v):
    # obtained with sympy
    res = (-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * (par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)\
          * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (u*v*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(u)*np.log(v))
    return res


def _ll(par, u, v):
    # obtained with sympy
    res = np.log((-np.log(u))**par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
                 * (par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1)
                 * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))
                 / (u*v*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(u)*np.log(v)))
    return res


def _d_ll_d_par(par, u, v):
    # obtained with sympy
    res = (-par**3*(-np.log(u))**par*np.log(-np.log(u)) + par**3*(-np.log(u))**par*np.log(-np.log(v))
           + par**3*(-np.log(v))**par*np.log(-np.log(u)) - par**3*(-np.log(v))**par*np.log(-np.log(v))
           - 2*par**2*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))
           + par**2*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))
           + 2*par**2*(-np.log(u))**par*np.log(-np.log(u)) - par**2*(-np.log(u))**par*np.log(-np.log(v))
           + par**2*(-np.log(u))**par + par**2*(-np.log(v))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))
           - 2*par**2*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))
           - par**2*(-np.log(v))**par*np.log(-np.log(u)) + 2*par**2*(-np.log(v))**par*np.log(-np.log(v))
           + par**2*(-np.log(v))**par - par*(-np.log(u))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(u))
           + par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
           * np.log((-np.log(u))**par + (-np.log(v))**par) + 3*par*(-np.log(u))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))
           - par*(-np.log(u))**par*np.log((-np.log(u))**par + (-np.log(v))**par)
           - par*(-np.log(u))**par*np.log(-np.log(u)) - par*(-np.log(v))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log(-np.log(v)) + par*(-np.log(v))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)
           + 3*par*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))
           - par*(-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par)
           - par*(-np.log(v))**par*np.log(-np.log(v)) + (-np.log(u))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)
           - 3*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
           * np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(u))**par
           * np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(v))**par
           * ((-np.log(u))**par + (-np.log(v))**par)**(2/par)*np.log((-np.log(u))**par + (-np.log(v))**par)
           - 3*(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
           * np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(v))**par
           * np.log((-np.log(u))**par + (-np.log(v))**par))\
          / (par**2*((-np.log(u))**par + (-np.log(v))**par)
             * (par + ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)) - 1))
    return res


def _d_cdf_d_par(par, u, v):
    # obtained with sympy
    res = -((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * (((-np.log(u))**par*np.log(-np.log(u)) + (-np.log(v))**par*np.log(-np.log(v)))
             / (par*((-np.log(u))**par + (-np.log(v))**par))
             - np.log((-np.log(u))**par + (-np.log(v))**par)/par**2)\
          * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))
    return res


def _d_hfun_d_par(par, u, v):
    # obtained with sympy
    res = -(-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * (-par**2*(-np.log(u))**par*np.log(-np.log(u)) + par**2*(-np.log(u))**par*np.log(-np.log(v))
             - par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))
             + par*(-np.log(u))**par*np.log(-np.log(u)) - par*(-np.log(v))**par
             * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))
             + par*(-np.log(v))**par*np.log(-np.log(v)) + (-np.log(u))**par
             * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
             * np.log((-np.log(u))**par + (-np.log(v))**par) - (-np.log(u))**par
             * np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(v))**par
             * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log((-np.log(u))**par + (-np.log(v))**par)
             - (-np.log(v))**par*np.log((-np.log(u))**par + (-np.log(v))**par))\
          * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (par**2*v*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(v))
    return res


def _d_vfun_d_par(par, u, v):
    # obtained with sympy
    res = -(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * (par**2*(-np.log(v))**par*np.log(-np.log(u)) - par**2*(-np.log(v))**par*np.log(-np.log(v))
             - par*(-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(u))
             + par*(-np.log(u))**par*np.log(-np.log(u)) - par*(-np.log(v))**par
             * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))*np.log(-np.log(v))
             + par*(-np.log(v))**par*np.log(-np.log(v)) + (-np.log(u))**par
             * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
             * np.log((-np.log(u))**par + (-np.log(v))**par) - (-np.log(u))**par
             * np.log((-np.log(u))**par + (-np.log(v))**par) + (-np.log(v))**par
             * ((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
             * np.log((-np.log(u))**par + (-np.log(v))**par) - (-np.log(v))**par
             * np.log((-np.log(u))**par + (-np.log(v))**par))\
          * np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (par**2*u*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(u))
    return res


def _d_hfun_d_v(par, u, v):
    # obtained with sympy
    res = (-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * (-par*(-np.log(u))**par + (-np.log(u))**par*np.log(v) + (-np.log(u))**par
             + (-np.log(v))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
             + (-np.log(v))**par*np.log(v))*np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (v**2*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(v)**2)
    return res


def _d_vfun_d_u(par, u, v):
    # obtained with sympy
    res = (-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))\
          * (-par*(-np.log(v))**par + (-np.log(u))**par*((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0))
             + (-np.log(u))**par*np.log(u) + (-np.log(v))**par*np.log(u)
             + (-np.log(v))**par)*np.exp(-((-np.log(u))**par + (-np.log(v))**par)**(par**(-1.0)))\
          / (u**2*((-np.log(u))**par + (-np.log(v))**par)**2*np.log(u)**2)
    return res


gumbel_cop_funs = {'cdf': _cdf,
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
