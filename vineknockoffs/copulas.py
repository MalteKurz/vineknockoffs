import numpy as np
from abc import ABC, abstractmethod

import scipy.integrate as integrate
from scipy.optimize import fmin_l_bfgs_b, root_scalar
from scipy.stats import norm, kendalltau

from ._utils_copula_families import clayton_cop_funs, frank_cop_funs, gaussian_cop_funs, gumbel_cop_funs, indep_cop_funs


class Copula(ABC):
    _theta_bounds = None
    n_pars = np.nan
    trim_thres = 1e-12

    def __init__(self, par, cop_funs, rotation=0):
        self._par = par
        self._cop_funs = cop_funs
        self._rotation = rotation

    def __repr__(self):
        return f'{self.__class__.__name__}(par={self.par}, rotation={self.rotation})'

    @property
    def par(self):
        return self._par

    @property
    def rotation(self):
        return self._rotation

    def mle_est(self, u, v):
        tau, _ = kendalltau(u, v)
        theta_0 = self.tau2par(tau)
        theta_hat, _, _ = fmin_l_bfgs_b(self._neg_ll,
                                        theta_0,
                                        self._neg_ll_d_theta,
                                        (u, v),
                                        bounds=self._theta_bounds)
        self._par = theta_hat
        return

    @abstractmethod
    def tau2par(self, tau):
        pass

    def _trim_obs(self, u):
        u[u < self.trim_thres] = self.trim_thres
        u[u > 1. - self.trim_thres] = 1. - self.trim_thres
        return u

    def cdf(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['cdf'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = u - self._cop_funs['cdf'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = u + v - 1 + self._cop_funs['cdf'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = v - self._cop_funs['cdf'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def pdf(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['pdf'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['pdf'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['pdf'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['pdf'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def ll(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['ll'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['ll'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['ll'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['ll'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def _neg_ll(self, theta, u, v):
        if self.rotation == 0:
            res = -np.sum(self._cop_funs['ll'](theta, self._trim_obs(u), self._trim_obs(v)))
        elif self.rotation == 90:
            res = -np.sum(self._cop_funs['ll'](theta, self._trim_obs(1.-v), self._trim_obs(u)))
        elif self.rotation == 180:
            res = -np.sum(self._cop_funs['ll'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v)))
        else:
            assert self.rotation == 270
            res = -np.sum(self._cop_funs['ll'](theta, self._trim_obs(v), self._trim_obs(1.-u)))
        return res

    def _neg_ll_d_theta(self, theta, u, v):
        if self.rotation == 0:
            res = -np.sum(self._cop_funs['d_ll_d_theta'](theta, self._trim_obs(u), self._trim_obs(v)))
        elif self.rotation == 90:
            res = -np.sum(self._cop_funs['d_ll_d_theta'](theta, self._trim_obs(1.-v), self._trim_obs(u)))
        elif self.rotation == 180:
            res = -np.sum(self._cop_funs['d_ll_d_theta'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v)))
        else:
            assert self.rotation == 270
            res = -np.sum(self._cop_funs['d_ll_d_theta'](theta, self._trim_obs(v), self._trim_obs(1.-u)))
        return res

    def aic(self, u, v):
        res = 2 * self.n_pars + 2 * self._neg_ll(self.par, self._trim_obs(u), self._trim_obs(v))
        return res

    def hfun(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['hfun'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['vfun'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = 1. - self._cop_funs['hfun'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = 1. - self._cop_funs['vfun'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return self._trim_obs(res)

    def vfun(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['vfun'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = 1. - self._cop_funs['hfun'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = 1. - self._cop_funs['vfun'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['hfun'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return self._trim_obs(res)

    def d_hfun_d_theta(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_hfun_d_theta'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['d_vfun_d_theta'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = -1. * self._cop_funs['d_hfun_d_theta'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -1. * self._cop_funs['d_vfun_d_theta'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_vfun_d_theta(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_vfun_d_theta'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -1. * self._cop_funs['d_hfun_d_theta'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = -1. * self._cop_funs['d_vfun_d_theta'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['d_hfun_d_theta'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_hfun_d_v(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_hfun_d_v'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -1. * self._cop_funs['d_vfun_d_u'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['d_hfun_d_v'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -1. * self._cop_funs['d_vfun_d_u'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_vfun_d_u(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_vfun_d_u'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -1. * self._cop_funs['d_hfun_d_v'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['d_vfun_d_u'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -1. * self._cop_funs['d_hfun_d_v'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def inv_hfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        kwargs = {'bracket': [1e-12, 1-1e-12], 'method': 'brentq', 'xtol': 1e-12, 'rtol': 1e-12}

        if self.rotation == 0:
            res = np.array([root_scalar(lambda xx: self._cop_funs['hfun'](self.par, xx, v[i]) - u[i],
                                        **kwargs).root for i in range(len(u))])
        elif self.rotation == 90:
            res = np.array([root_scalar(lambda xx: self._cop_funs['vfun'](self.par, (1 - v[i]), xx) - u[i],
                                        **kwargs).root for i in range(len(u))])
        elif self.rotation == 180:
            res = 1. - np.array([root_scalar(lambda xx: self._cop_funs['hfun'](self.par, xx, 1. - v[i]) - (1. - u[i]),
                                             **kwargs).root for i in range(len(u))])
        else:
            assert self.rotation == 270
            res = 1. - np.array([root_scalar(lambda xx: self._cop_funs['vfun'](self.par, v[i], xx) - (1. - u[i]),
                                             **kwargs).root for i in range(len(u))])

        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        kwargs = {'bracket': [1e-12, 1-1e-12], 'method': 'brentq', 'xtol': 1e-12, 'rtol': 1e-12}

        if self.rotation == 0:
            res = np.array([root_scalar(lambda xx: self._cop_funs['vfun'](self.par, u[i], xx) - v[i],
                                        **kwargs).root for i in range(len(v))])
        elif self.rotation == 90:
            res = 1. - np.array([root_scalar(lambda xx: self._cop_funs['hfun'](self.par, xx, u[i]) - (1. - v[i]),
                                             **kwargs).root for i in range(len(v))])
        elif self.rotation == 180:
            res = 1. - np.array([root_scalar(lambda xx: self._cop_funs['vfun'](self.par, 1. - u[i], xx) - (1. - v[i]),
                                             **kwargs).root for i in range(len(v))])
        else:
            assert self.rotation == 270
            res = np.array([root_scalar(lambda xx: self._cop_funs['hfun'](self.par, xx, (1. - u[i])) - v[i],
                                        **kwargs).root for i in range(len(v))])

        return self._trim_obs(res)

    def d_hfun_d_u(self, u, v):
        return self.pdf(u, v)

    def d_vfun_d_v(self, u, v):
        return self.pdf(u, v)

    def d_inv_hfun_d_u(self, u, v):
        return 1. / self.pdf(self.inv_hfun(u, v), v)

    def d_inv_vfun_d_v(self, u, v):
        return 1. / self.pdf(u, self.inv_vfun(u, v))

    def d_inv_hfun_d_v(self, u, v):
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_v(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_u(self, u, v):
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_u(u, xx) / self.pdf(u, xx)

    def d_inv_hfun_d_theta(self, u, v):
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_theta(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_theta(self, u, v):
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_theta(u, xx) / self.pdf(u, xx)

    def sim(self, n_obs=100):
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_hfun(u[:, 0], u[:, 1])
        return u


class ClaytonCopula(Copula):
    n_pars = 1

    def __init__(self, par=None, rotation=0):
        super().__init__(par, clayton_cop_funs, rotation=rotation)
        self._theta_bounds = [(0.0001, 28)]

    def tau2par(self, tau):
        if self.rotation in [0, 180]:
            par = 2 * tau / (1 - tau)
        else:
            assert self.rotation in [90, 270]
            tau *= -1.
            par = 2 * tau / (1 - tau)
        return par


class FrankCopula(Copula):
    n_pars = 1

    def __init__(self, par=None):
        super().__init__(par, frank_cop_funs)
        self._theta_bounds = [(-40, 40)]

    def par2tau(self, theta):
        # ToDO: Check and compare with R
        debye_fun = integrate.quad(lambda x: x / np.expm1(x), 0, theta)[0]
        tau = 1 - 4/theta*(1-debye_fun/theta)
        return tau

    def tau2par(self, tau):
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
    n_pars = 1

    def __init__(self, par=None):
        super().__init__(par, gaussian_cop_funs)
        self._theta_bounds = [(-0.999, 0.999)]

    def tau2par(self, tau):
        return np.sin(np.pi * tau / 2)

    def inv_hfun(self, u, v):
        res = self._cop_funs['inv_hfun'](self.par, u, v)
        return res

    def inv_vfun(self, u, v):
        res = self._cop_funs['inv_vfun'](self.par, u, v)
        return res


class GumbelCopula(Copula):
    n_pars = 1

    def __init__(self, par=None, rotation=0):
        super().__init__(par, gumbel_cop_funs, rotation=rotation)
        self._theta_bounds = [(1.0, 20)]

    def tau2par(self, tau):
        if self.rotation in [0, 180]:
            par = 1/(1 - tau)
        else:
            assert self.rotation in [90, 270]
            tau *= -1.
            par = 1/(1 - tau)
        return par


class IndepCopula(Copula):
    n_pars = 0

    def __init__(self):
        super().__init__(None, indep_cop_funs)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def tau2par(self, tau):
        return None

    def mle_est(self, u, v):
        return None


def cop_select(u, v, families='all', rotations=True, indep_test=True):
    assert families == 'all'
    copulas = [ClaytonCopula(), FrankCopula(), GumbelCopula(), GaussianCopula()]
    if rotations:
        tau, _ = kendalltau(u, v)
        if tau >= 0.:
            rots = [180]
        else:
            rots = [90, 270]
        copulas += [ClaytonCopula(rotation=rot) for rot in rots]
        copulas += [GumbelCopula(rotation=rot) for rot in rots]
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
