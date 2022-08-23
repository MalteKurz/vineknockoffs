import numpy as np
from abc import ABC, abstractmethod

import warnings

import scipy.integrate as integrate
from scipy.optimize import fmin_l_bfgs_b, root_scalar
from scipy.stats import norm, kendalltau

from ._copula_familes.clayton import clayton_cop_funs
from ._copula_familes.frank import frank_cop_funs
from ._copula_familes.gaussian import gaussian_cop_funs
from ._copula_familes.gumbel import gumbel_cop_funs
from ._copula_familes.indep import indep_cop_funs


class Copula(ABC):
    n_pars = 0
    trim_thres = 1e-12

    def __init__(self, par, cop_funs, par_bounds, rotation=0):
        self._par = np.full(self.n_pars, np.nan)
        self._par_bounds = par_bounds
        self.par = par
        self._cop_funs = cop_funs
        self._rotation = rotation

    def __repr__(self):
        return f'{self.__class__.__name__}(par={self.par})'

    @property
    def par(self):
        return self._par

    @par.setter
    def par(self, value):
        if np.isscalar(value):
            value = np.array([value])
        if len(value) != self.n_pars:
            raise ValueError(f'{self.n_pars} parameters expected but {len(value)} provided')
        for i_par in range(self.n_pars):
            if (value[i_par] < self._par_bounds[i_par][0]) | (value[i_par] > self._par_bounds[i_par][1]):
                raise ValueError(f'The {i_par+1}. parameter must be in '
                                 f'[{self._par_bounds[i_par][0]}, {self._par_bounds[i_par][1]}]')
            else:
                self._par[i_par] = value[i_par]

    def set_par_w_bound_check(self, value, tol=1e-4):
        if np.isscalar(value):
            value = np.array([value])
        if len(value) != self.n_pars:
            raise ValueError(f'{self.n_pars} parameters expected but {len(value)} provided')
        for i_par in range(self.n_pars):
            if value[i_par] < self._par_bounds[i_par][0]:
                self._par[i_par] = self._par_bounds[i_par][0] + tol
                warnings.warn(f'{i_par+1}. parameter of {self.__class__.__name__} copula adjusted '
                              f'from {value[i_par]} to {self._par[i_par]}')
            elif value[i_par] > self._par_bounds[i_par][1]:
                self._par[i_par] = self._par_bounds[i_par][1] - tol
                warnings.warn(f'{i_par+1}. parameter of {self.__class__.__name__} copula adjusted '
                              f'from {value[i_par]} to {self._par[i_par]}')
            else:
                self._par[i_par] = value[i_par]
        return self

    @property
    def rotation(self):
        return self._rotation

    def mle_est(self, u, v):
        tau, _ = kendalltau(u, v)
        par_0 = self.tau2par(tau)
        par_hat, _, _ = fmin_l_bfgs_b(self._neg_ll,
                                      par_0,
                                      self._neg_ll_d_par,
                                      (u, v),
                                      bounds=self._par_bounds)
        self._par = par_hat
        return

    @abstractmethod
    def tau2par(self, tau):
        pass

    @abstractmethod
    def par2tau(self, par):
        pass

    def _trim_obs(self, u):
        u = np.array(u)
        u[u < self.trim_thres] = self.trim_thres
        u[u > 1. - self.trim_thres] = 1. - self.trim_thres
        return u

    def _cdf(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['cdf'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = u - self._cop_funs['cdf'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = u + v - 1 + self._cop_funs['cdf'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = v - self._cop_funs['cdf'](par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def cdf(self, u, v):
        return self._cdf(par=self.par, u=u, v=v)

    def _d_cdf_d_par(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_cdf_d_par'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -self._cop_funs['d_cdf_d_par'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['d_cdf_d_par'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -self._cop_funs['d_cdf_d_par'](par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_cdf_d_par(self, u, v):
        return self._d_cdf_d_par(par=self.par, u=u, v=v)

    def _pdf_cc(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['pdf'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['pdf'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['pdf'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['pdf'](par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def _pdf(self, par, u, v):
        res = self._pdf_cc(par, u, v)
        return res

    def pdf(self, u, v):
        return self._pdf(par=self.par, u=u, v=v)

    def _ll(self, par, u, v):
        return np.log(self._pdf(par=par, u=u, v=v))

    def ll(self, u, v):
        return self._ll(par=self.par, u=u, v=v)

    def _neg_ll(self, par, u, v):
        return -np.sum(self._ll(par=par, u=u, v=v))

    def _neg_ll_d_par_cc(self, par, u, v):
        if self.rotation == 0:
            res = -np.sum(self._cop_funs['d_ll_d_par'](par, self._trim_obs(u), self._trim_obs(v)))
        elif self.rotation == 90:
            res = -np.sum(self._cop_funs['d_ll_d_par'](par, self._trim_obs(1.-v), self._trim_obs(u)))
        elif self.rotation == 180:
            res = -np.sum(self._cop_funs['d_ll_d_par'](par, self._trim_obs(1.-u), self._trim_obs(1.-v)))
        else:
            assert self.rotation == 270
            res = -np.sum(self._cop_funs['d_ll_d_par'](par, self._trim_obs(v), self._trim_obs(1.-u)))
        return res

    def _neg_ll_d_par(self, par, u, v):
        res = self._neg_ll_d_par_cc(par, u, v)
        return res

    def aic(self, u, v):
        res = 2 * self.n_pars + 2 * self._neg_ll(self.par, u=u, v=v)
        return res

    def _hfun(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['hfun'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['vfun'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = 1. - self._cop_funs['hfun'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = 1. - self._cop_funs['vfun'](par, self._trim_obs(v), self._trim_obs(1.-u))

        return self._trim_obs(res)

    def hfun(self, u, v):
        return self._hfun(par=self.par, u=u, v=v)

    def _vfun(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['vfun'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = 1. - self._cop_funs['hfun'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = 1. - self._cop_funs['vfun'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['hfun'](par, self._trim_obs(v), self._trim_obs(1.-u))
        return self._trim_obs(res)

    def vfun(self, u, v, u_=None):
        return self._vfun(par=self.par, u=u, v=v)

    def _d_hfun_d_par(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_hfun_d_par'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['d_vfun_d_par'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = -1. * self._cop_funs['d_hfun_d_par'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -1. * self._cop_funs['d_vfun_d_par'](par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_hfun_d_par(self, u, v):
        return self._d_hfun_d_par(par=self.par, u=u, v=v)

    def _d_vfun_d_par(self, par, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_vfun_d_par'](par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -1. * self._cop_funs['d_hfun_d_par'](par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = -1. * self._cop_funs['d_vfun_d_par'](par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['d_hfun_d_par'](par, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_vfun_d_par(self, u, v):
        return self._d_vfun_d_par(par=self.par, u=u, v=v)

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

        res = np.array([root_scalar(lambda xx: self._hfun(par=self.par, u=xx, v=v[i]) - u[i],
                                    **kwargs).root for i in range(len(u))])

        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        kwargs = {'bracket': [1e-12, 1-1e-12], 'method': 'brentq', 'xtol': 1e-12, 'rtol': 1e-12}

        res = np.array([root_scalar(lambda xx: self._vfun(par=self.par, u=u[i], v=xx) - v[i],
                                    **kwargs).root for i in range(len(u))])

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

    def d_inv_hfun_d_par(self, u, v):
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_par(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_par(self, u, v):
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_par(u, xx) / self.pdf(u, xx)

    def sim(self, n_obs=100):
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_hfun(u[:, 0], u[:, 1])
        return u


class ClaytonCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan, rotation=0):
        super().__init__(par, clayton_cop_funs,
                         par_bounds=[(0.0001, 28)], rotation=rotation)

    def __repr__(self):
        return f'{self.__class__.__name__}(par={self.par}, rotation={self.rotation})'

    def tau2par(self, tau):
        if self.rotation in [0, 180]:
            par = 2 * tau / (1 - tau)
        else:
            assert self.rotation in [90, 270]
            tau *= -1.
            par = 2 * tau / (1 - tau)
        return par

    def par2tau(self, par):
        if self.rotation in [0, 180]:
            tau = par / (par + 2)
        else:
            assert self.rotation in [90, 270]
            tau = par / (par + 2)
            tau *= -1.
        return tau

    def inv_hfun(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['inv_hfun'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['inv_vfun'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = 1. - self._cop_funs['inv_hfun'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = 1. - self._cop_funs['inv_vfun'](self.par, self._trim_obs(v), self._trim_obs(1.-u))

        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        if self.rotation == 0:
            res = self._cop_funs['inv_vfun'](self.par, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = 1. - self._cop_funs['inv_hfun'](self.par, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = 1. - self._cop_funs['inv_vfun'](self.par, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['inv_hfun'](self.par, self._trim_obs(v), self._trim_obs(1.-u))
        return self._trim_obs(res)


class FrankCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan):
        super().__init__(par, frank_cop_funs,
                         par_bounds=[(-40, 40)])

    def par2tau(self, par):
        # ToDO: Check and compare with R
        debye_fun = integrate.quad(lambda x: x / np.expm1(x), 0, par)[0]
        tau = 1 - 4/par*(1-debye_fun/par)
        return tau

    def tau2par(self, tau):
        # ToDO: Check and compare with R
        tau_l = FrankCopula().par2tau(-40)
        tau_u = FrankCopula().par2tau(40)
        if (tau < tau_l) | (tau > tau_u):
            raise ValueError(f'Choose Kendall tau between {tau_l} and {tau_u}.')
        if tau == 0.:
            par = 0.
        else:
            if tau > 0:
                bracket = [0.0001, 40]
            else:
                bracket = [-40, -0.0001]
            root_res = root_scalar(lambda xx: FrankCopula().par2tau(xx) - tau,
                                   bracket=bracket,
                                   method='brentq')
            par = root_res.root
        return par

    def inv_hfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        res = self._cop_funs['inv_hfun'](self.par, u, v)
        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        res = self._cop_funs['inv_vfun'](self.par, u, v)
        return self._trim_obs(res)


class GaussianCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan):
        super().__init__(par, gaussian_cop_funs,
                         par_bounds=[(-0.999, 0.999)])

    def tau2par(self, tau):
        return np.sin(np.pi * tau / 2)

    def par2tau(self, par):
        return 2/np.pi * np.arcsin(par)

    def inv_hfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        res = self._cop_funs['inv_hfun'](self.par, u, v)
        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        res = self._cop_funs['inv_vfun'](self.par, u, v)
        return self._trim_obs(res)


class GumbelCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan, rotation=0):
        super().__init__(par, gumbel_cop_funs,
                         par_bounds=[(1.0, 20)], rotation=rotation)

    def __repr__(self):
        return f'{self.__class__.__name__}(par={self.par}, rotation={self.rotation})'

    def tau2par(self, tau):
        if self.rotation in [0, 180]:
            par = 1/(1 - tau)
        else:
            assert self.rotation in [90, 270]
            tau *= -1.
            par = 1/(1 - tau)
        return par

    def par2tau(self, par):
        if self.rotation in [0, 180]:
            tau = 1 - 1/par
        else:
            assert self.rotation in [90, 270]
            tau = 1 - 1/par
            tau *= -1.
        return tau


class IndepCopula(Copula):
    n_pars = 0

    def __init__(self):
        super().__init__(np.array([]), indep_cop_funs,
                         par_bounds=[])

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def tau2par(self, tau):
        return None

    def par2tau(self, par):
        return None

    def mle_est(self, u, v):
        return None

    def inv_hfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        res = self._cop_funs['inv_hfun'](self.par, u, v)
        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        res = self._cop_funs['inv_vfun'](self.par, u, v)
        return self._trim_obs(res)


def cop_select(u, v, families='all', rotations=True, indep_test=True):
    assert families == 'all'
    copulas = [cop()
               for cop in [FrankCopula, GaussianCopula]]
    if rotations:
        tau, _ = kendalltau(u, v)
        if tau >= 0.:
            rots = [0, 180]
        else:
            rots = [90, 270]
    else:
        rots = [0]
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
