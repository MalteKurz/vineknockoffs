import numpy as np
from abc import ABC, abstractmethod

import warnings

import scipy.integrate as integrate
from scipy.optimize import fmin_l_bfgs_b, root_scalar
from scipy.stats import norm, kendalltau

from ._utils_copula_families import clayton_cop_funs, frank_cop_funs, gaussian_cop_funs, gumbel_cop_funs, indep_cop_funs


class Copula(ABC):
    n_pars = 0
    trim_thres = 1e-12

    def __init__(self, par, cop_funs, theta_bounds, rotation=0, u_discrete=False, v_discrete=False):
        self._par = np.full(self.n_pars, np.nan)
        self._theta_bounds = theta_bounds
        self.par = par
        self._cop_funs = cop_funs
        self._rotation = rotation
        self._u_discrete = u_discrete
        self._v_discrete = v_discrete

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
            if (value[i_par] < self._theta_bounds[i_par][0]) | (value[i_par] > self._theta_bounds[i_par][1]):
                raise ValueError(f'The {i_par+1}. parameter must be in '
                                 f'[{self._theta_bounds[i_par][0]}, {self._theta_bounds[i_par][1]}]')
            else:
                self._par[i_par] = value[i_par]

    def set_par_w_bound_check(self, value, tol=1e-4):
        if np.isscalar(value):
            value = np.array([value])
        if len(value) != self.n_pars:
            raise ValueError(f'{self.n_pars} parameters expected but {len(value)} provided')
        for i_par in range(self.n_pars):
            if value[i_par] < self._theta_bounds[i_par][0]:
                self._par[i_par] = self._theta_bounds[i_par][0] + tol
                warnings.warn(f'{i_par+1}. parameter of {self.__class__.__name__} copula adjusted '
                              f'from {value[i_par]} to {self._par[i_par]}')
            elif value[i_par] > self._theta_bounds[i_par][1]:
                self._par[i_par] = self._theta_bounds[i_par][1] - tol
                warnings.warn(f'{i_par+1}. parameter of {self.__class__.__name__} copula adjusted '
                              f'from {value[i_par]} to {self._par[i_par]}')
            else:
                self._par[i_par] = value[i_par]
        return self

    @property
    def rotation(self):
        return self._rotation

    @property
    def u_discrete(self):
        return self._u_discrete

    @property
    def continuous_vars(self):
        return not (self.u_discrete | self.u_discrete)

    @property
    def v_discrete(self):
        return self._v_discrete

    def mle_est(self, u, v, u_=None, v_=None):
        tau, _ = kendalltau(u, v)
        theta_0 = self.tau2par(tau)
        theta_hat, _, _ = fmin_l_bfgs_b(self._neg_ll,
                                        theta_0,
                                        self._neg_ll_d_theta,
                                        (u, v, u_, v_),
                                        bounds=self._theta_bounds)
        self._par = theta_hat
        return

    @abstractmethod
    def tau2par(self, tau):
        pass

    @abstractmethod
    def par2tau(self, par):
        pass

    def _trim_obs(self, u):
        u[u < self.trim_thres] = self.trim_thres
        u[u > 1. - self.trim_thres] = 1. - self.trim_thres
        return u

    def _cdf(self, theta, u, v):
        if self.rotation == 0:
            res = self._cop_funs['cdf'](theta, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = u - self._cop_funs['cdf'](theta, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = u + v - 1 + self._cop_funs['cdf'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = v - self._cop_funs['cdf'](theta, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def cdf(self, u, v):
        return self._cdf(theta=self.par, u=u, v=v)

    def _d_cdf_d_theta(self, theta, u, v):
        if self.rotation == 0:
            res = self._cop_funs['d_cdf_d_theta'](theta, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -self._cop_funs['d_cdf_d_theta'](theta, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['d_cdf_d_theta'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -self._cop_funs['d_cdf_d_theta'](theta, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_cdf_d_theta(self, u, v):
        return self._d_cdf_d_theta(theta=self.par, u=u, v=v)

    def _pdf_cc(self, theta, u, v):
        if self.rotation == 0:
            res = self._cop_funs['pdf'](theta, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['pdf'](theta, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = self._cop_funs['pdf'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['pdf'](theta, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def _pdf_dc(self, theta, u, v, u_):
        assert not self.v_discrete
        return self._hfun(theta, u=u, v=v) - self._hfun(theta, u=u_, v=v)

    def _pdf_cd(self, theta, u, v, v_):
        assert not self.u_discrete
        return self._vfun(theta, u=u, v=v) - self._vfun(theta, u=u, v=v_)

    def _pdf_dd(self, theta, u, v, u_, v_):
        return self._cdf(theta, u, v) - self._cdf(theta, u_, v) - self._cdf(theta, u, v_) + self._cdf(theta, u_, v_)

    def _pdf(self, theta, u, v, u_=None, v_=None):
        if self.continuous_vars:
            res = self._pdf_cc(theta, u, v)
        elif self.u_discrete & (not self.v_discrete):
            res = self._pdf_dc(theta, u, v, u_=u_)
        elif (not self.u_discrete) & self.v_discrete:
            res = self._pdf_cd(theta, u, v, v_=v_)
        else:
            assert self.u_discrete & self.v_discrete
            res = self._pdf_dd(theta, u, v, u_=u_, v_=v_)

        return res

    def pdf(self, u, v, u_=None, v_=None):
        return self._pdf(theta=self.par, u=u, v=v, u_=u_, v_=v_)

    def _ll(self, theta, u, v, u_=None, v_=None):
        return np.log(self._pdf(theta=theta, u=u, v=v, u_=u_, v_=v_))

    def ll(self, u, v, u_=None, v_=None):
        return self._ll(theta=self.par, u=u, v=v, u_=u_, v_=v_)

    def _neg_ll(self, theta, u, v, u_=None, v_=None):
        return -np.sum(self._ll(theta=theta, u=u, v=v, u_=u_, v_=v_))

    def _neg_ll_d_theta_cc(self, theta, u, v):
        return -np.sum(self._cop_funs['d_ll_d_theta'](theta, self._trim_obs(u), self._trim_obs(v)))

    def _neg_ll_d_theta_dc(self, theta, u, v, u_):
        nom = self._d_hfun_d_theta(theta, u, v) - self._d_hfun_d_theta(theta, u_, v)
        denom = self._pdf_dc(theta=theta, u=u, v=v, u_=u_)
        return -np.sum(nom / denom)

    def _neg_ll_d_theta_cd(self, theta, u, v, v_):
        nom = self._d_vfun_d_theta(theta, u, v) - self._d_vfun_d_theta(theta, u, v_)
        denom = self._pdf_cd(theta=theta, u=u, v=v, v_=v_)
        return -np.sum(nom / denom)

    def _neg_ll_d_theta_dd(self, theta, u, v, u_, v_):
        nom = self._d_cdf_d_theta(theta, u, v) - self._d_cdf_d_theta(theta, u_, v) \
            - self._d_cdf_d_theta(theta, u, v_) + self._d_cdf_d_theta(theta, u_, v_)
        denom = self._pdf_dd(theta=theta, u=u, v=v, u_=u_, v_=v_)
        return -np.sum(nom / denom)

    def _neg_ll_d_theta(self, theta, u, v, u_=None, v_=None):
        if self.continuous_vars:
            res = self._neg_ll_d_theta_cc(theta, u, v)
        elif self.u_discrete & (not self.v_discrete):
            res = self._neg_ll_d_theta_dc(theta, u, v, u_=u_)
        elif (not self.u_discrete) & self.v_discrete:
            res = self._neg_ll_d_theta_cd(theta, u, v, v_=v_)
        else:
            assert self.u_discrete & self.v_discrete
            res = self._neg_ll_d_theta_dd(theta, u, v, u_=u_, v_=v_)

        return res

    def aic(self, u, v, u_=None, v_=None):
        res = 2 * self.n_pars + 2 * self._neg_ll(self.par, u=u, v=v, u_=u_, v_=v_)
        return res

    def _hfun(self, theta, u, v, v_=None):
        if self.rotation == 0:
            if not self.v_discrete:
                res = self._cop_funs['hfun'](theta, self._trim_obs(u), self._trim_obs(v))
            else:
                res = (self._cdf(theta, u, v) - self._cdf(theta, u, v_)) / (v - v_)
        elif self.rotation == 90:
            if not self.v_discrete:
                res = self._cop_funs['vfun'](theta, self._trim_obs(1.-v), self._trim_obs(u))
            else:
                res = (self._cdf(theta, 1.-v, u) - self._cdf(theta, 1.-v_, u)) / (v_ - v)
        elif self.rotation == 180:
            if not self.v_discrete:
                res = 1. - self._cop_funs['hfun'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
            else:
                res = 1. - (self._cdf(theta, 1.-u, 1.-v) - self._cdf(theta, 1.-u, 1.-v_)) / (v_ - v)
        else:
            assert self.rotation == 270
            if not self.v_discrete:
                res = 1. - self._cop_funs['vfun'](theta, self._trim_obs(v), self._trim_obs(1.-u))
            else:
                res = 1. - (self._cdf(theta, v, 1.-u) - self._cdf(theta, v_, 1.-u)) / (v - v_)

        return self._trim_obs(res)

    def hfun(self, u, v, v_=None):
        return self._hfun(theta=self.par, u=u, v=v, v_=v_)

    def _vfun(self, theta, u, v, u_=None):
        if self.rotation == 0:
            if not self.u_discrete:
                res = self._cop_funs['vfun'](theta, self._trim_obs(u), self._trim_obs(v))
            else:
                res = (self._cdf(theta, u, v) - self._cdf(theta, u_, v)) / (u - u_)
        elif self.rotation == 90:
            if not self.u_discrete:
                res = 1. - self._cop_funs['hfun'](theta, self._trim_obs(1.-v), self._trim_obs(u))
            else:
                res = 1. - (self._cdf(theta, 1.-v, u) - self._cdf(theta, 1.-v, u_)) / (u - u_)
        elif self.rotation == 180:
            if not self.u_discrete:
                res = 1. - self._cop_funs['vfun'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
            else:
                res = 1. - (self._cdf(theta, 1.-u, 1.-v) - self._cdf(theta, 1.-u_, 1.-v)) / (u_ - u)
        else:
            assert self.rotation == 270
            if not self.u_discrete:
                res = self._cop_funs['hfun'](theta, self._trim_obs(v), self._trim_obs(1.-u))
            else:
                res = (self._cdf(theta, v, 1.-u) - self._cdf(theta, v, 1.-u_)) / (u_ - u)
        return self._trim_obs(res)

    def vfun(self, u, v, u_=None):
        return self._vfun(theta=self.par, u=u, v=v, u_=u_)

    def _d_hfun_d_theta(self, theta, u, v):
        assert not self.v_discrete
        if self.rotation == 0:
            res = self._cop_funs['d_hfun_d_theta'](theta, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = self._cop_funs['d_vfun_d_theta'](theta, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = -1. * self._cop_funs['d_hfun_d_theta'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = -1. * self._cop_funs['d_vfun_d_theta'](theta, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_hfun_d_theta(self, u, v):
        return self._d_hfun_d_theta(theta=self.par, u=u, v=v)

    def _d_vfun_d_theta(self, theta, u, v):
        assert not self.u_discrete
        if self.rotation == 0:
            res = self._cop_funs['d_vfun_d_theta'](theta, self._trim_obs(u), self._trim_obs(v))
        elif self.rotation == 90:
            res = -1. * self._cop_funs['d_hfun_d_theta'](theta, self._trim_obs(1.-v), self._trim_obs(u))
        elif self.rotation == 180:
            res = -1. * self._cop_funs['d_vfun_d_theta'](theta, self._trim_obs(1.-u), self._trim_obs(1.-v))
        else:
            assert self.rotation == 270
            res = self._cop_funs['d_hfun_d_theta'](theta, self._trim_obs(v), self._trim_obs(1.-u))
        return res

    def d_vfun_d_theta(self, u, v):
        return self._d_vfun_d_theta(theta=self.par, u=u, v=v)

    def d_hfun_d_v(self, u, v):
        assert self.continuous_vars
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
        assert self.continuous_vars
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
        assert self.continuous_vars
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
        assert self.continuous_vars
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
        assert self.continuous_vars
        return self.pdf(u, v)

    def d_vfun_d_v(self, u, v):
        assert self.continuous_vars
        return self.pdf(u, v)

    def d_inv_hfun_d_u(self, u, v):
        assert self.continuous_vars
        return 1. / self.pdf(self.inv_hfun(u, v), v)

    def d_inv_vfun_d_v(self, u, v):
        assert self.continuous_vars
        return 1. / self.pdf(u, self.inv_vfun(u, v))

    def d_inv_hfun_d_v(self, u, v):
        assert self.continuous_vars
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_v(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_u(self, u, v):
        assert self.continuous_vars
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_u(u, xx) / self.pdf(u, xx)

    def d_inv_hfun_d_theta(self, u, v):
        assert self.continuous_vars
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_theta(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_theta(self, u, v):
        assert self.continuous_vars
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_theta(u, xx) / self.pdf(u, xx)

    def sim(self, n_obs=100):
        assert self.continuous_vars
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_hfun(u[:, 0], u[:, 1])
        return u


class ClaytonCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan, rotation=0, u_discrete=False, v_discrete=False):
        super().__init__(par, clayton_cop_funs,
                         theta_bounds=[(0.0001, 28)], rotation=rotation,
                         u_discrete=u_discrete, v_discrete=v_discrete)

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


class FrankCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan, u_discrete=False, v_discrete=False):
        super().__init__(par, frank_cop_funs,
                         theta_bounds=[(-40, 40)],
                         u_discrete=u_discrete, v_discrete=v_discrete)

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

    def __init__(self, par=np.nan, u_discrete=False, v_discrete=False):
        super().__init__(par, gaussian_cop_funs,
                         theta_bounds=[(-0.999, 0.999)],
                         u_discrete=u_discrete, v_discrete=v_discrete)

    def tau2par(self, tau):
        return np.sin(np.pi * tau / 2)

    def par2tau(self, par):
        return 2/np.pi * np.arcsin(par)

    def inv_hfun(self, u, v):
        assert self.continuous_vars
        res = self._cop_funs['inv_hfun'](self.par, u, v)
        return res

    def inv_vfun(self, u, v):
        assert self.continuous_vars
        res = self._cop_funs['inv_vfun'](self.par, u, v)
        return res


class GumbelCopula(Copula):
    n_pars = 1

    def __init__(self, par=np.nan, rotation=0, u_discrete=False, v_discrete=False):
        super().__init__(par, gumbel_cop_funs,
                         theta_bounds=[(1.0, 20)], rotation=rotation,
                         u_discrete=u_discrete, v_discrete=v_discrete)

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

    def __init__(self, u_discrete=False, v_discrete=False):
        super().__init__(np.array([]), indep_cop_funs,
                         theta_bounds=[],
                         u_discrete=u_discrete, v_discrete=v_discrete)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def tau2par(self, tau):
        return None

    def par2tau(self, theta):
        return None

    def mle_est(self, u, v, u_=None, v_=None):
        return None


def cop_select(u, v, families='all', rotations=True, indep_test=True,
               u_=None, v_=None, u_discrete=False, v_discrete=False):
    assert families == 'all'
    if u_ is None:
        u_ = np.full_like(u, np.nan)
    if v_ is None:
        v_ = np.full_like(v, np.nan)
    copulas = [cop(u_discrete=u_discrete, v_discrete=v_discrete)
               for cop in [FrankCopula, GaussianCopula]]
    if rotations:
        tau, _ = kendalltau(u, v)
        if tau >= 0.:
            rots = [0, 180]
        else:
            rots = [90, 270]
    else:
        rots = [0]
    copulas += [ClaytonCopula(u_discrete=u_discrete, v_discrete=v_discrete, rotation=rot) for rot in rots]
    copulas += [GumbelCopula(u_discrete=u_discrete, v_discrete=v_discrete, rotation=rot) for rot in rots]
    indep_cop = False
    if indep_test:
        n_obs = len(u)
        tau, _ = kendalltau(u, v)
        # ToDo check whether the independence test is valid for discrete variables
        test_stat = np.sqrt(9*n_obs*(n_obs-1)/2/(2*n_obs+5)) * np.abs(tau)
        indep_cop = (test_stat <= norm.ppf(0.975))

    if indep_cop:
        cop_sel = IndepCopula(u_discrete=u_discrete, v_discrete=v_discrete)
    else:
        aics = np.full(len(copulas), np.nan)
        for ind, this_cop in enumerate(copulas):
            this_cop.mle_est(u, v, u_=u_, v_=v_)
            aics[ind] = this_cop.aic(u, v, u_=u_, v_=v_)
        best_ind = np.argmin(aics)
        cop_sel = copulas[best_ind]

    return cop_sel
