import numpy as np
from abc import ABC, abstractmethod

from scipy.stats import kendalltau
from scipy.optimize import fmin_l_bfgs_b, root_scalar

from sympy.utilities.lambdify import lambdify
from sympy.codegen.rewriting import optimize, optims_c99
from sympy import diff, log


def opt_and_lambdify(fun, u_sym, v_sym, theta_sym):
    fun = optimize(fun, optims_c99)
    ufun = lambdify((theta_sym, u_sym, v_sym),
                    fun,
                    'numpy')
    return fun, ufun


def copula_derivs_one_par(cdf_sym, u_sym, v_sym, theta_sym):
    ufuns = dict()

    cdf_sym, ufuns['cdf'] = opt_and_lambdify(cdf_sym,
                                             u_sym, v_sym, theta_sym)

    hfun_sym = diff(cdf_sym, v_sym)
    hfun_sym, ufuns['hfun'] = opt_and_lambdify(hfun_sym,
                                               u_sym, v_sym, theta_sym)

    vfun_sym = diff(cdf_sym, u_sym)
    vfun_sym, ufuns['vfun'] = opt_and_lambdify(vfun_sym,
                                               u_sym, v_sym, theta_sym)

    pdf_sym = diff(hfun_sym, u_sym)
    pdf_sym, ufuns['pdf'] = opt_and_lambdify(pdf_sym,
                                             u_sym, v_sym, theta_sym)

    ll_sym = log(pdf_sym)
    ll_sym, ufuns['ll'] = opt_and_lambdify(ll_sym,
                                           u_sym, v_sym, theta_sym)

    d_ll_d_theta_sym = diff(ll_sym, theta_sym)
    d_ll_d_theta_sym, ufuns['d_ll_d_theta'] = opt_and_lambdify(d_ll_d_theta_sym,
                                                               u_sym, v_sym, theta_sym)

    d_hfun_d_theta_sym = diff(hfun_sym, theta_sym)
    d_hfun_d_theta_sym, ufuns['d_hfun_d_theta'] = opt_and_lambdify(d_hfun_d_theta_sym,
                                                                   u_sym, v_sym, theta_sym)

    d_vfun_d_theta_sym = diff(vfun_sym, theta_sym)
    d_vfun_d_theta_sym, ufuns['d_vfun_d_theta'] = opt_and_lambdify(d_vfun_d_theta_sym,
                                                                   u_sym, v_sym, theta_sym)

    d_hfun_d_v_sym = diff(hfun_sym, v_sym)
    d_hfun_d_v_sym, ufuns['d_hfun_d_v'] = opt_and_lambdify(d_hfun_d_v_sym,
                                                           u_sym, v_sym, theta_sym)

    d_vfun_d_u_sym = diff(vfun_sym, u_sym)
    d_vfun_d_u_sym, ufuns['d_vfun_d_u'] = opt_and_lambdify(d_vfun_d_u_sym,
                                                           u_sym, v_sym, theta_sym)

    return ufuns


class Copula(ABC):
    _theta_bounds = None
    n_par = np.nan
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
        res = 2 * self.n_par + 2 * self._neg_ll(self.par, self._trim_obs(u), self._trim_obs(v))
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

    def sim(self, n_obs=100):
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_hfun(u[:, 0], u[:, 1])
        return u
