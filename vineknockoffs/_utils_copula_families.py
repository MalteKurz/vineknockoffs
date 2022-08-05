import numpy as np

from sympy import symbols
# from sympy import diff, log, exp, sqrt

from scipy.stats import norm, multivariate_normal

from ._utils_copulas import opt_and_lambdify, read_and_lambdify_sympy_expr
# from ._utils_copulas import sym_copula_derivs_one_par, write_sympy_expr, copula_derivs_one_par

u_sym, v_sym, par_sym = symbols('u v par')

# gumbel_cdf_sym = exp(-((-log(u_sym))**par_sym + (-log(v_sym))**par_sym)**(1/par_sym))
# gumbel_cop_funs = copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, par_sym)
# gumbel_sym_dict = sym_copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(gumbel_sym_dict, './vineknockoffs/sym_copula_expr/gumbel.csv')

gumbel_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'gumbel.csv',
                                               (par_sym, u_sym, v_sym))

# clayton_cdf_sym = (u_sym**(-par_sym) + v_sym**(-par_sym) - 1)**(-1/par_sym)
# clayton_cop_funs = copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, par_sym)
# clayton_sym_dict = sym_copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(clayton_sym_dict, './vineknockoffs/sym_copula_expr/clayton.csv')

clayton_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'clayton.csv',
                                                (par_sym, u_sym, v_sym))

# frank_cdf_sym = - 1/par_sym * \
#                 log(1/(1 - exp(-par_sym)) *
#                     (1 - exp(-par_sym) - (1 - exp(-par_sym*u_sym)) * (1 - exp(-par_sym*v_sym))))
# frank_cop_funs = copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, par_sym)
# frank_sym_dict = sym_copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, par_sym)
# write_sympy_expr(frank_sym_dict, './vineknockoffs/sym_copula_expr/frank.csv')

frank_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'frank.csv',
                                              (par_sym, u_sym, v_sym))

x_sym, y_sym, par_sym = symbols('x y par')

# gauss_cop_xy_funs = dict()
# gauss_sym_dict = dict()
# gauss_pdf_sym = 1/(sqrt(1-par_sym**2)) * exp(-(par_sym**2*(x_sym**2 + y_sym**2) - 2*par_sym*x_sym*y_sym)
#                                                / (2*(1-par_sym**2)))
# gauss_sym_dict['pdf'], gauss_cop_xy_funs['pdf'] = opt_and_lambdify(gauss_pdf_sym, x_sym, y_sym, par_sym)
#
# gauss_ll_sym = log(1/(sqrt(1-par_sym**2))) - (par_sym**2*(x_sym**2 + y_sym**2)
#                                                 - 2*par_sym*x_sym*y_sym) / (2*(1-par_sym**2))
# gauss_sym_dict['ll'], gauss_cop_xy_funs['ll'] = opt_and_lambdify(gauss_ll_sym, x_sym, y_sym, par_sym)
#
# gauss_d_ll_d_par_sym = diff(gauss_ll_sym, par_sym)
# gauss_sym_dict['d_ll_d_par'], gauss_cop_xy_funs['d_ll_d_par'] = opt_and_lambdify(gauss_d_ll_d_par_sym,
#                                                                                      x_sym, y_sym, par_sym)
# write_sympy_expr(gauss_sym_dict, './vineknockoffs/sym_copula_expr/gaussian.csv')

gauss_cop_xy_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'gaussian.csv',
                                                 (par_sym, x_sym, y_sym))


def gaussian_cdf(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    return multivariate_normal.cdf(np.column_stack((x, y)),
                                   mean=[0., 0.],
                                   cov=[[1., par], [par, 1.]])


def gaussian_pdf(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = gauss_cop_xy_funs['pdf'](par, x, y)
    return res


def gaussian_ll(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = gauss_cop_xy_funs['ll'](par, x, y)
    return res


def gaussian_d_ll_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = gauss_cop_xy_funs['d_ll_d_par'](par, x, y)
    return res


def gaussian_d_cdf_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = 1 / (2 * np.pi * np.sqrt(1 - par**2)) * np.exp((2 * par * x * y - x**2 - y**2) / (2 * (1 - par**2)))
    return res


def gaussian_hfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf((x - par * y) / np.sqrt(1 - par ** 2))
    return res


def gaussian_vfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf((y - par * x) / np.sqrt(1 - par ** 2))
    return res


def gaussian_d_hfun_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    xx = 1 - par ** 2
    res = norm.pdf((x - par * y) / np.sqrt(xx))
    res *= (-y * np.sqrt(xx) + (x - par * y) * par / np.sqrt(xx)) / xx
    return res


def gaussian_d_vfun_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    xx = 1 - par ** 2
    res = norm.pdf((y - par * x) / np.sqrt(xx))
    res *= (-x * np.sqrt(xx) + (y - par * x) * par / np.sqrt(xx)) / xx
    return res


def gaussian_d_hfun_d_v(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    d_y_d_v = np.sqrt(2 * np.pi) * np.exp(y ** 2 / 2)
    xx = 1 - par ** 2
    res = norm.pdf((x - par * y) / np.sqrt(xx))
    res *= -par / np.sqrt(xx) * d_y_d_v
    return res


def gaussian_d_vfun_d_u(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    d_x_d_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
    xx = 1 - par ** 2
    res = norm.pdf((y - par * x) / np.sqrt(xx))
    res *= -par / np.sqrt(xx) * d_x_d_u
    return res


def gaussian_inv_hfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf(x * np.sqrt(1 - par ** 2) + par * y)
    return res


def gaussian_inv_vfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf(y * np.sqrt(1 - par ** 2) + par * x)
    return res


gaussian_cop_funs = {'cdf': gaussian_cdf,
                     'pdf': gaussian_pdf,
                     'll': gaussian_ll,
                     'd_ll_d_par': gaussian_d_ll_d_par,
                     'd_cdf_d_par': gaussian_d_cdf_d_par,
                     'hfun': gaussian_hfun,
                     'vfun': gaussian_vfun,
                     'd_hfun_d_par': gaussian_d_hfun_d_par,
                     'd_vfun_d_par': gaussian_d_vfun_d_par,
                     'd_hfun_d_v': gaussian_d_hfun_d_v,
                     'd_vfun_d_u': gaussian_d_vfun_d_u,
                     'inv_hfun': gaussian_inv_hfun,
                     'inv_vfun': gaussian_inv_vfun,
                     }

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
