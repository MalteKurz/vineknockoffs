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

    def __init__(self, par, cop_funs):
        self._par = par
        self._cop_funs = cop_funs

    @property
    def par(self):
        return self._par

    def mle_est(self, u, v):
        tau, _ = kendalltau(u, v)
        theta_0 = self.tau2par(tau)
        theta_hat, _, _ = fmin_l_bfgs_b(self._neg_ll,
                                        theta_0,
                                        self._neg_ll_deriv_theta,
                                        (u, v),
                                        bounds=self._theta_bounds)
        self._par = theta_hat
        return

    @staticmethod
    @abstractmethod
    def tau2par(tau):
        pass

    def cdf(self, u, v):
        return self._cop_funs['cdf'](self.par, u, v)

    def pdf(self, u, v):
        return self._cop_funs['pdf'](self.par, u, v)

    def ll(self, u, v):
        return self._cop_funs['ll'](self.par, u, v)

    def _neg_ll(self, theta, u, v):
        return -np.sum(self._cop_funs['ll'](theta, u, v))

    def _neg_ll_deriv_theta(self, theta, u, v):
        return -np.sum(self._cop_funs['d_ll_d_theta'](theta, u, v))

    def aic(self, u, v):
        res = 2 * self.n_par + 2 * self._neg_ll(self.par, u, v)
        return res

    def hfun(self, u, v):
        res = self._cop_funs['hfun'](self.par, u, v)
        return res

    def vfun(self, u, v):
        res = self._cop_funs['vfun'](self.par, u, v)
        return res

    def d_hfun_d_theta(self, u, v):
        res = self._cop_funs['d_hfun_d_theta'](self.par, u, v)
        return res

    def d_vfun_d_theta(self, u, v):
        res = self._cop_funs['d_vfun_d_theta'](self.par, u, v)
        return res

    def d_hfun_d_v(self, u, v):
        res = self._cop_funs['d_hfun_d_v'](self.par, u, v)
        return res

    def d_vfun_d_u(self, u, v):
        res = self._cop_funs['d_vfun_d_u'](self.par, u, v)
        return res

    def inv_h_fun(self, u, v):
        res = np.array([root_scalar(lambda xx: self._cop_funs['hfun'](self.par, xx, v[i]) - u[i],
                                    bracket=[1e-12, 1-1e-12],
                                    method='brentq',
                                    xtol=1e-12, rtol=1e-12).root for i in range(len(u))])
        return res

    def inv_v_fun(self, u, v):
        res = np.array([root_scalar(lambda xx: self._cop_funs['vfun'](self.par, xx, u[i]) - v[i],
                                    bracket=[1e-12, 1-1e-12],
                                    method='brentq',
                                    xtol=1e-12, rtol=1e-12).root for i in range(len(v))])
        return res

    def sim(self, n_obs=100):
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_h_fun(u[:, 0], u[:, 1])
        return u
