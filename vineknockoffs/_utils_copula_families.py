import numpy as np

from sympy import symbols, diff, log, exp, sqrt

import scipy.integrate as integrate
from scipy.optimize import root_scalar
from scipy.stats import norm, multivariate_normal, kendalltau

from ._utils_copula import Copula, copula_derivs_one_par, opt_and_lambdify
from ._utils_gaussian_copula import gaussian_cop_funs

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

indep_cdf_sym = u_sym * v_sym
indep_cop_funs = copula_derivs_one_par(indep_cdf_sym, u_sym, v_sym, theta_sym)


class ClaytonCopula(Copula):
    n_par = 1

    def __init__(self, par=None):
        super().__init__(par, clayton_cop_funs)
        self._theta_bounds = [(0.0001, 28)]

    @staticmethod
    def tau2par(tau):
        return 2 * tau / (1 - tau)


class FrankCopula(Copula):
    n_par = 1

    def __init__(self, par=None):
        super().__init__(par, frank_cop_funs)
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
    n_par = 1

    def __init__(self, par=None):
        super().__init__(par, gaussian_cop_funs)
        self._theta_bounds = [(-0.999, 0.999)]

    @staticmethod
    def tau2par(tau):
        return np.sin(np.pi * tau / 2)

    def inv_h_fun(self, u, v):
        res = self._cop_funs['inv_h_fun'](self.par, u, v)
        return res

    def inv_v_fun(self, u, v):
        res = self._cop_funs['inv_v_fun'](self.par, u, v)
        return res


class GumbelCopula(Copula):
    n_par = 1

    def __init__(self, par=None):
        super().__init__(par, gumbel_cop_funs)
        self._theta_bounds = [(1.0, 20)]

    @staticmethod
    def tau2par(tau):
        return 1/(1 - tau)


class IndepCopula(Copula):
    n_par = 1

    def __init__(self):
        super().__init__(None, indep_cop_funs)

    @staticmethod
    def tau2par(tau):
        return None

    def mle_est(self, u, v):
        return None


def cop_select(u, v, families='all', indep_test=True):
    assert families == 'all'
    copulas = [ClaytonCopula(), FrankCopula(), GumbelCopula(), GaussianCopula()]
    indep_cop = False
    if indep_test:
        n_obs = len(u)
        tau = kendalltau(u, v)
        test_stat = np.sqrt(9*n_obs*(n_obs-1)/2/(2*n_obs+5)) * np.abs(tau)
        indep_cop = (test_stat <= norm.ppf(0.975))

    if indep_cop:
        cop_sel = IndepCopula()
    else:
        aics = np.full(len(copulas), np.nan)
        for this_cop in copulas:
            par_hat = this_cop.mle_est(u, v)
