import numpy as np

from sympy import symbols, log, exp

import scipy.integrate as integrate
from scipy.optimize import root_scalar

from ._utils_copula import Copula, copula_derivs_one_par

u_sym, v_sym, theta_sym = symbols('u v theta')

gumbel_cdf_sym = exp(-((-log(u_sym))**theta_sym + (-log(v_sym))**theta_sym)**(1/theta_sym))
gumbel_cop_funs = copula_derivs_one_par(gumbel_cdf_sym, u_sym, v_sym, theta_sym)

clayton_cdf_sym = (u_sym**(-theta_sym) + v_sym**(-theta_sym) - 1)**(-1/theta_sym)
clayton_cop_funs = copula_derivs_one_par(clayton_cdf_sym, u_sym, v_sym, theta_sym)

frank_cdf_sym = - 1/theta_sym * \
                log(1/(1 - exp(-theta_sym)) *
                    (1 - exp(-theta_sym) - (1 - exp(-theta_sym*u_sym)) * (1 - exp(-theta_sym*v_sym))))
frank_cop_funs = copula_derivs_one_par(frank_cdf_sym, u_sym, v_sym, theta_sym)


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


class GumbelCopula(Copula):

    def __init__(self):
        super().__init__(gumbel_cop_funs)
        self._theta_bounds = [(1.0, 20)]

    @staticmethod
    def tau2par(tau):
        return 1/(1 - tau)
