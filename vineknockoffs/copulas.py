import numpy as np

import scipy.integrate as integrate
from scipy.optimize import root_scalar
from scipy.stats import norm, kendalltau

from ._utils_copulas import Copula
from ._utils_copula_families import clayton_cop_funs, frank_cop_funs, gaussian_cop_funs, gumbel_cop_funs, indep_cop_funs


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
        tau, _ = kendalltau(u, v)
        test_stat = np.sqrt(9*n_obs*(n_obs-1)/2/(2*n_obs+5)) * np.abs(tau)
        indep_cop = (test_stat <= norm.ppf(0.975))

    if indep_cop:
        cop_sel = IndepCopula()
    else:
        aics = np.full(len(copulas), np.nan)
        for ind, this_cop in enumerate(copulas):
            this_cop.mle_est(u, v)
            aics[ind] = this_cop.aic(u, v)
        best_ind = np.argmin(aics)
        cop_sel = copulas[best_ind]

    return cop_sel
