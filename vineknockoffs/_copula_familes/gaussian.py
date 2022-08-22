import numpy as np
from scipy.stats import norm, multivariate_normal


def _pdf_xy(par, x, y):
    # obtained with sympy
    res = np.exp((-par**2*(x**2 + y**2) + 2*par*x*y)/(2 - 2*par**2))/np.sqrt(1 - par**2)
    return res


def _ll_xy(par, x, y):
    # obtained with sympy
    res = -np.log(-(par - 1)*(par + 1))/2 - (par**2*(x**2 + y**2) - 2*par*x*y)/(2 - 2*par**2)
    return res


def _d_ll_d_par_xy(par, x, y):
    # obtained with sympy
    res = -(par**3 - par**2*x*y + par*x**2 + par*y**2 - par - x*y)/((par - 1)**2*(par + 1)**2)
    return res


def _cdf(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    return multivariate_normal.cdf(np.column_stack((x, y)),
                                   mean=[0., 0.],
                                   cov=[[1., par], [par, 1.]])


def _pdf(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = _pdf_xy(par, x, y)
    return res


def _ll(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = _ll_xy(par, x, y)
    return res


def _d_ll_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = _d_ll_d_par_xy(par, x, y)
    return res


def _d_cdf_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = 1 / (2 * np.pi * np.sqrt(1 - par**2)) * np.exp((2 * par * x * y - x**2 - y**2) / (2 * (1 - par**2)))
    return res


def _hfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf((x - par * y) / np.sqrt(1 - par ** 2))
    return res


def _vfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf((y - par * x) / np.sqrt(1 - par ** 2))
    return res


def _d_hfun_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    xx = 1 - par ** 2
    res = norm.pdf((x - par * y) / np.sqrt(xx))
    res *= (-y * np.sqrt(xx) + (x - par * y) * par / np.sqrt(xx)) / xx
    return res


def _d_vfun_d_par(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    xx = 1 - par ** 2
    res = norm.pdf((y - par * x) / np.sqrt(xx))
    res *= (-x * np.sqrt(xx) + (y - par * x) * par / np.sqrt(xx)) / xx
    return res


def _d_hfun_d_v(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    d_y_d_v = np.sqrt(2 * np.pi) * np.exp(y ** 2 / 2)
    xx = 1 - par ** 2
    res = norm.pdf((x - par * y) / np.sqrt(xx))
    res *= -par / np.sqrt(xx) * d_y_d_v
    return res


def _d_vfun_d_u(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    d_x_d_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
    xx = 1 - par ** 2
    res = norm.pdf((y - par * x) / np.sqrt(xx))
    res *= -par / np.sqrt(xx) * d_x_d_u
    return res


def _inv_hfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf(x * np.sqrt(1 - par ** 2) + par * y)
    return res


def _inv_vfun(par, u, v):
    x = norm.ppf(u)
    y = norm.ppf(v)
    res = norm.cdf(y * np.sqrt(1 - par ** 2) + par * x)
    return res


gaussian_cop_funs = {'cdf': _cdf,
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
