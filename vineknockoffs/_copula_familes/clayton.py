import numpy as np

# from ._utils_copula_families_sympy import sym_copula_derivs_one_par, write_sympy_expr, copula_derivs_one_par
# from sympy import diff, log, exp, sqrt
# u_sym, v_sym, par_sym = symbols('u v par')

# clayton_cdf_sym = (u_sym**(-par_sym) + v_sym**(-par_sym) - 1)**(-1/par_sym)
# clayton_cop_funs = copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, par_sym)
# clayton_sym_dict = sym_copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(clayton_sym_dict, './clayton.csv')


def _cdf(par, u, v):
    # obtained with sympy
    res = (-1 + v**(-par) + u**(-par))**(-1/par)
    return res


def _hfun(par, u, v):
    # obtained with sympy
    res = v**(-par)*(-1 + v**(-par) + u**(-par))**(-1/par)/(v*(-1 + v**(-par) + u**(-par)))
    return res


def _vfun(par, u, v):
    # obtained with sympy
    res = u**(-par)*(-1 + v**(-par) + u**(-par))**(-1/par)/(u*(-1 + v**(-par) + u**(-par)))
    return res


def _pdf(par, u, v):
    # obtained with sympy
    res = u**par*v**par*(par + 1)*(-1 + v**(-par) + u**(-par))**(-1/par)/(u*v*(u**par*v**par - u**par - v**par)**2)
    return res


def _ll(par, u, v):
    # obtained with sympy
    res = np.log(u**par*v**par*(par + 1)*(-1 + v**(-par) + u**(-par))**(-1/par) /
                 (u*v*(u**par*v**par - u**par - v**par)**2))
    return res


def _d_ll_d_par(par, u, v):
    # obtained with sympy
    res = (-par**3*u**par*v**par*np.log(u) - par**3*u**par*v**par*np.log(v) + par**3*u**par*np.log(u)
           - par**3*u**par*np.log(v) - par**3*v**par*np.log(u) + par**3*v**par*np.log(v)
           - par**2*u**par*v**par*np.log(u) - par**2*u**par*v**par*np.log(v) + par**2*u**par*v**par
           + par**2*u**par*np.log(u) - 2*par**2*u**par*np.log(v) - par**2*u**par - 2*par**2*v**par*np.log(u)
           + par**2*v**par*np.log(v) - par**2*v**par + par*u**par*v**par *
           np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - par*u**par*np.log(v)
           - par*u**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)) - par*v**par*np.log(u)
           - par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
           + u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
           - u**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
           - v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)))\
          /(par**2*(par + 1)*(u**par*v**par - u**par - v**par))
    return res


def _d_cdf_d_par(par, u, v):
    # obtained with sympy
    res = (-(-v**(-par)*np.log(v) - u**(-par)*np.log(u))/(par*(-1 + v**(-par) + u**(-par)))
           + np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))/par**2)\
          * (-1 + v**(-par) + u**(-par))**(-1/par)
    return res


def _d_hfun_d_par(par, u, v):
    # obtained with sympy
    res = -u**par*(-1 + v**(-par) + u**(-par))**(-1/par)\
          * (-par**2*u**par*v**par*np.log(v) - par**2*v**par*np.log(u) + par**2*v**par*np.log(v) - par*u**par*np.log(v)
             - par*v**par*np.log(u) + u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
             - u**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
             - v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)))\
          / (par**2*v*(u**par*v**par - u**par - v**par)**2)
    return res


def _d_vfun_d_par(par, u, v):
    # obtained with sympy
    res = -v**par*(-1 + v**(-par) + u**(-par))**(-1/par)\
          * (-par**2*u**par*v**par*np.log(u) + par**2*u**par*np.log(u) - par**2*u**par*np.log(v) - par*u**par*np.log(v)
             - par*v**par*np.log(u) + u**par*v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
             - u**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par))
             - v**par*np.log(-u**(-par)*v**(-par)*(u**par*v**par - u**par - v**par)))\
          / (par**2*u*(u**par*v**par - u**par - v**par)**2)
    return res


def _d_hfun_d_v(par, u, v):
    # obtained with sympy
    res = u**par*v**par*(par + 1)*(u**par - 1)*(-1 + v**(-par) + u**(-par))**(-1/par)\
          / (v**2*(u**par*v**par - u**par - v**par)**2)
    return res


def _d_vfun_d_u(par, u, v):
    # obtained with sympy
    res = u**par*v**par*(par + 1)*(v**par - 1)*(-1 + v**(-par) + u**(-par))**(-1/par)\
          / (u**2*(u**par*v**par - u**par - v**par)**2)
    return res


def _inv_hfun(par, u, v):
    h1 = -1 / par
    h2 = -par / (1 + par)
    res = np.power(np.power(v, -par) * (np.power(u, h2) - 1) + 1, h1)
    return res


def _inv_vfun(par, u, v):
    h1 = -1 / par
    h2 = -par / (1 + par)
    res = np.power(np.power(u, -par) * (np.power(v, h2) - 1) + 1, h1)
    return res


clayton_cop_funs = {'cdf': _cdf,
                    'pdf': _pdf,
                    'll': _ll,
                    'd_ll_d_par': _d_ll_d_par,
                    'd_cdf_d_par': _d_cdf_d_par,
                    'hfun': _hfun,
                    'vfun': _vfun,
                    'd_hfun_d_par': _d_hfun_d_par,
                    'd_vfun_d_par': _d_vfun_d_par,
                    'd_hfun_d_v': _d_hfun_d_v,
                    'd_vfun_d_u': _d_vfun_d_u,
                    'inv_hfun': _inv_hfun,
                    'inv_vfun': _inv_vfun,
                    }
