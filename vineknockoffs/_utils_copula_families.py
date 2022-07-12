import numpy as np

from sympy import symbols
# from sympy import diff, log, exp, sqrt

from scipy.stats import norm, multivariate_normal

from ._utils_copulas import opt_and_lambdify, read_and_lambdify_sympy_expr
# from ._utils_copulas import sym_copula_derivs_one_par, write_sympy_expr, copula_derivs_one_par

u_sym, v_sym, theta_sym = symbols('u v theta')

# gumbel_cdf_sym = exp(-((-log(u_sym))**theta_sym + (-log(v_sym))**theta_sym)**(1/theta_sym))
# gumbel_cop_funs = copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, theta_sym)
# gumbel_sym_dict = sym_copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, theta_sym)
# write_sympy_expr(gumbel_sym_dict, './vineknockoffs/sym_copula_expr/gumbel.csv')

gumbel_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'gumbel.csv',
                                               (theta_sym, u_sym, v_sym))

# clayton_cdf_sym = (u_sym**(-theta_sym) + v_sym**(-theta_sym) - 1)**(-1/theta_sym)
# clayton_cop_funs = copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, theta_sym)
# clayton_sym_dict = sym_copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, theta_sym)
# write_sympy_expr(clayton_sym_dict, './vineknockoffs/sym_copula_expr/clayton.csv')

clayton_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'clayton.csv',
                                                (theta_sym, u_sym, v_sym))

# frank_cdf_sym = - 1/theta_sym * \
#                 log(1/(1 - exp(-theta_sym)) *
#                     (1 - exp(-theta_sym) - (1 - exp(-theta_sym*u_sym)) * (1 - exp(-theta_sym*v_sym))))
# frank_cop_funs = copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, theta_sym)
# frank_sym_dict = sym_copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, theta_sym)
# write_sympy_expr(frank_sym_dict, './vineknockoffs/sym_copula_expr/frank.csv')

frank_cop_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'frank.csv',
                                              (theta_sym, u_sym, v_sym))

x_sym, y_sym, theta_sym = symbols('x y theta')

# gauss_cop_xy_funs = dict()
# gauss_sym_dict = dict()
# gauss_pdf_sym = 1/(sqrt(1-theta_sym**2)) * exp(-(theta_sym**2*(x_sym**2 + y_sym**2) - 2*theta_sym*x_sym*y_sym)
#                                                / (2*(1-theta_sym**2)))
# gauss_sym_dict['pdf'], gauss_cop_xy_funs['pdf'] = opt_and_lambdify(gauss_pdf_sym, x_sym, y_sym, theta_sym)
#
# gauss_ll_sym = log(1/(sqrt(1-theta_sym**2))) - (theta_sym**2*(x_sym**2 + y_sym**2)
#                                                 - 2*theta_sym*x_sym*y_sym) / (2*(1-theta_sym**2))
# gauss_sym_dict['ll'], gauss_cop_xy_funs['ll'] = opt_and_lambdify(gauss_ll_sym, x_sym, y_sym, theta_sym)
#
# gauss_d_ll_d_theta_sym = diff(gauss_ll_sym, theta_sym)
# gauss_sym_dict['d_ll_d_theta'], gauss_cop_xy_funs['d_ll_d_theta'] = opt_and_lambdify(gauss_d_ll_d_theta_sym,
#                                                                                      x_sym, y_sym, theta_sym)
# write_sympy_expr(gauss_sym_dict, './vineknockoffs/sym_copula_expr/gaussian.csv')

gauss_cop_xy_funs = read_and_lambdify_sympy_expr('vineknockoffs.sym_copula_expr', 'gaussian.csv',
                                                 (theta_sym, x_sym, y_sym))


def gaussian_cdf(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    return multivariate_normal.cdf(np.column_stack((x, y)),
                                   mean=[0., 0.],
                                   cov=[[1., theta], [theta, 1.]])


def gaussian_pdf(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = gauss_cop_xy_funs['pdf'](theta, x, y)
    return res


def gaussian_ll(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = gauss_cop_xy_funs['ll'](theta, x, y)
    return res


def gaussian_d_ll_d_theta(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = gauss_cop_xy_funs['d_ll_d_theta'](theta, x, y)
    return res


def gaussian_hfun(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf((x - theta * y) / np.sqrt(1 - theta ** 2))
    return res


def gaussian_vfun(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf((y - theta * x) / np.sqrt(1 - theta ** 2))
    return res


def gaussian_d_hfun_d_theta(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    xx = 1 - theta ** 2
    res = norm.pdf((x - theta * y) / np.sqrt(xx))
    res *= (-y * np.sqrt(xx) + (x - theta * y) * theta / np.sqrt(xx)) / xx
    return res


def gaussian_d_vfun_d_theta(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    xx = 1 - theta ** 2
    res = norm.pdf((y - theta * x) / np.sqrt(xx))
    res *= (-x * np.sqrt(xx) + (y - theta * x) * theta / np.sqrt(xx)) / xx
    return res


def gaussian_d_hfun_d_v(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    d_y_d_v = np.sqrt(2 * np.pi) * np.exp(y ** 2 / 2)
    xx = 1 - theta ** 2
    res = norm.pdf((x - theta * y) / np.sqrt(xx))
    res *= -theta / np.sqrt(xx) * d_y_d_v
    return res


def gaussian_d_vfun_d_u(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    d_x_d_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
    xx = 1 - theta ** 2
    res = norm.pdf((y - theta * x) / np.sqrt(xx))
    res *= -theta / np.sqrt(xx) * d_x_d_u
    return res


def gaussian_inv_hfun(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf(x * np.sqrt(1 - theta ** 2) + theta * y)
    return res


def gaussian_inv_vfun(theta, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf(y * np.sqrt(1 - theta ** 2) + theta * x)
    return res


gaussian_cop_funs = {'cdf': gaussian_cdf,
                     'pdf': gaussian_pdf,
                     'll': gaussian_ll,
                     'd_ll_d_theta': gaussian_d_ll_d_theta,
                     'hfun': gaussian_hfun,
                     'vfun': gaussian_vfun,
                     'd_hfun_d_theta': gaussian_d_hfun_d_theta,
                     'd_vfun_d_theta': gaussian_d_vfun_d_theta,
                     'd_hfun_d_v': gaussian_d_hfun_d_v,
                     'd_vfun_d_u': gaussian_d_vfun_d_u,
                     'inv_hfun': gaussian_inv_hfun,
                     'inv_vfun': gaussian_inv_vfun,
                     }

indep_cop_funs = {'cdf': lambda theta, u, v: u * v,
                  'pdf': lambda theta, u, v: np.ones_like(u),
                  'll': lambda theta, u, v: np.zeros_like(u),
                  'd_ll_d_theta': lambda theta, u, v: np.zeros_like(u),
                  'hfun': lambda theta, u, v: u,
                  'vfun': lambda theta, u, v: v,
                  'd_hfun_d_theta': lambda theta, u, v: np.zeros_like(u),
                  'd_vfun_d_theta': lambda theta, u, v: np.zeros_like(u),
                  'd_hfun_d_v': lambda theta, u, v: np.zeros_like(u),
                  'd_vfun_d_u': lambda theta, u, v: np.zeros_like(u),
                  'inv_hfun': lambda theta, u, v: u,
                  'inv_vfun': lambda theta, u, v: v,
                  }
