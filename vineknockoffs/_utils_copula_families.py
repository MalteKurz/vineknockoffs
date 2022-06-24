import numpy as np

from sympy import symbols, diff, log, exp, sqrt

import scipy.integrate as integrate
from scipy.optimize import root_scalar
from scipy.stats import norm, multivariate_normal

from ._utils_copula import Copula, copula_derivs_one_par, opt_and_lambdify

u_sym, v_sym, theta_sym = symbols('u v theta')

gumbel_cdf_sym = exp(-((-log(u_sym))**theta_sym + (-log(v_sym))**theta_sym)**(1/theta_sym))
gumbel_cop_funs = copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, theta_sym)

clayton_cdf_sym = (u_sym**(-theta_sym) + v_sym**(-theta_sym) - 1)**(-1/theta_sym)
clayton_cop_funs = copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, theta_sym)

frank_cdf_sym = - 1/theta_sym * \
                log(1/(1 - exp(-theta_sym)) *
                    (1 - exp(-theta_sym) - (1 - exp(-theta_sym*u_sym)) * (1 - exp(-theta_sym*v_sym))))
frank_cop_funs = copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, theta_sym)

x_sym, y_sym, theta_sym = symbols('x y theta')
gauss_cop_funs = dict()
gauss_pdf_sym = 1/(sqrt(1-theta_sym**2)) * exp(-(theta_sym**2*(x_sym**2 + y_sym**2) - 2*theta_sym*x_sym*y_sym)
                                               / (2*(1-theta_sym**2)))
gauss_pdf_sym, gauss_pdf_fun_xy = opt_and_lambdify(gauss_pdf_sym, x_sym, y_sym, theta_sym)

gauss_ll_sym = log(1/(sqrt(1-theta_sym**2))) - (theta_sym**2*(x_sym**2 + y_sym**2)
                                                - 2*theta_sym*x_sym*y_sym) / (2*(1-theta_sym**2))
gauss_ll_sym, gauss_ll_fun_xy = opt_and_lambdify(gauss_ll_sym, x_sym, y_sym, theta_sym)

gauss_d_ll_d_theta_sym = diff(gauss_ll_sym, theta_sym)
gauss_d_ll_d_theta_sym, gauss_d_ll_d_theta_fun_xy = opt_and_lambdify(gauss_d_ll_d_theta_sym, x_sym, y_sym, theta_sym)


class ClaytonCopula(Copula):

    def __init__(self):
        super().__init__(clayton_cop_funs)
        self._theta_bounds = [(0.0001, 28)]

    @staticmethod
    def tau2par(tau):
        return 2 * tau / (1 - tau)


class FrankCopula(Copula):

    def __init__(self):
        super().__init__(frank_cop_funs)
        self._theta_bounds = [(-40, 40)]

    @staticmethod
    def par2tau(theta):
        # ToDO: Check and compare with R
        debye_fun = integrate.quad(lambda x: x / np.expm1(x), 0, theta)[0]
        tau = 1 - 4/theta*(1-debye_fun/theta)
        return tau

    @staticmethod
    def tau2par(tau):
        # ToDO: Check and compare with R
        tau_l = FrankCopula().par2tau(-40)
        tau_u = FrankCopula().par2tau(40)
        if (tau < tau_l) | (tau > tau_u):
            raise ValueError(f'Choose Kendall tau between {tau_l} and {tau_u}.')
        if tau == 0.:
            theta = 0.
        else:
            if tau > 0:
                bracket = [0.0001, 40]
            else:
                bracket = [-40, -0.0001]
            root_res = root_scalar(lambda xx: FrankCopula().par2tau(xx) - tau,
                                   bracket=bracket,
                                   method='brentq')
            theta = root_res.root
        return theta


class GaussianCopula(Copula):

    def __init__(self):
        super().__init__(None)
        self._theta_bounds = [(-0.999, 0.999)]

    @staticmethod
    def tau2par(tau):
        return np.sin(np.pi * tau / 2)

    def cdf(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        return multivariate_normal.cdf(np.column_stack((x, y)),
                                       mean=[0., 0.],
                                       cov=[[1., theta], [theta, 1.]])

    def pdf(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = gauss_pdf_fun_xy(theta, x, y)
        return res

    def ll(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = gauss_ll_fun_xy(theta, x, y)
        return res

    def neg_ll_deriv_theta(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = -np.sum(gauss_d_ll_d_theta_fun_xy(theta, x, y))
        return res

    def hfun(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = norm.cdf((x - theta * y) / np.sqrt(1 - theta ** 2))
        return res

    def vfun(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = norm.cdf((y - theta * x) / np.sqrt(1 - theta ** 2))
        return res

    def d_hfun_d_theta(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        xx = 1 - theta ** 2
        res = norm.pdf((x - theta * y) / np.sqrt(xx))
        res *= (-y * np.sqrt(xx) + (x - theta * y) * theta / np.sqrt(xx)) / xx
        return res

    def d_vfun_d_theta(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        xx = 1 - theta ** 2
        res = norm.pdf((y - theta * x) / np.sqrt(xx))
        res *= (-x * np.sqrt(xx) + (y - theta * x) * theta / np.sqrt(xx)) / xx
        return res

    def d_hfun_d_v(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        d_y_d_v = np.sqrt(2 * np.pi) * np.exp(y ** 2 / 2)
        xx = 1 - theta ** 2
        res = norm.pdf((x - theta * y) / np.sqrt(xx))
        res *= -theta / np.sqrt(xx) * d_y_d_v
        return res

    def d_vfun_d_u(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        d_x_d_u = np.sqrt(2 * np.pi) * np.exp(x ** 2 / 2)
        xx = 1 - theta ** 2
        res = norm.pdf((y - theta * x) / np.sqrt(xx))
        res *= -theta / np.sqrt(xx) * d_x_d_u
        return res

    def inv_h_fun(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = norm.cdf(x * np.sqrt(1 - theta ** 2) + theta * y)
        return res

    def inv_v_fun(self, theta, u, v):
        x = norm.ppf(u)
        y = norm.ppf(v)
        res = norm.cdf(y * np.sqrt(1 - theta ** 2) + theta * x)
        return res


class GumbelCopula(Copula):

    def __init__(self):
        super().__init__(gumbel_cop_funs)
        self._theta_bounds = [(1.0, 20)]

    @staticmethod
    def tau2par(tau):
        return 1/(1 - tau)
