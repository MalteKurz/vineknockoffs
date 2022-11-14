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
    _trim_thres = 1e-12

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
        """
        Parameter of the copula.
        """
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
        """
        Rotation of the copula (0, 90, 180 or 270).
        """
        return self._rotation

    def mle_est(self, u, v):
        """
        Estimation of the copula parameter with maximum likelihood.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        self : object
        """
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
        u[u < self._trim_thres] = self._trim_thres
        u[u > 1. - self._trim_thres] = 1. - self._trim_thres
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
        """
        Evaluate the copula cdf.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the derivative of the copula cdf w.r.t. the parameter.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the copula pdf.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        return self._pdf(par=self.par, u=u, v=v)

    def _ll(self, par, u, v):
        return np.log(self._pdf(par=par, u=u, v=v))

    def ll(self, u, v):
        """
        Evaluate the copula log-likelihood.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the copula AIC.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the copula hfun.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the copula vfun.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the derivative of the copula hfun w.r.t. the parameter.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the derivative of the copula vfun w.r.t. the parameter.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        return self._d_vfun_d_par(par=self.par, u=u, v=v)

    def d_hfun_d_v(self, u, v):
        """
        Evaluate the derivative of the copula hfun w.r.t. v.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the derivative of the copula vfun w.r.t. u.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
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
        """
        Evaluate the inverse of the copula hfun.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        kwargs = {'bracket': [1e-12, 1-1e-12], 'method': 'brentq', 'xtol': 1e-12, 'rtol': 1e-12}

        res = np.array([root_scalar(lambda xx: self._hfun(par=self.par, u=xx, v=v[i]) - u[i],
                                    **kwargs).root for i in range(len(u))])

        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        """
        Evaluate the inverse of the copula vfun.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        kwargs = {'bracket': [1e-12, 1-1e-12], 'method': 'brentq', 'xtol': 1e-12, 'rtol': 1e-12}

        res = np.array([root_scalar(lambda xx: self._vfun(par=self.par, u=u[i], v=xx) - v[i],
                                    **kwargs).root for i in range(len(u))])

        return self._trim_obs(res)

    def d_hfun_d_u(self, u, v):
        """
        Evaluate the derivative of the copula hfun w.r.t. u.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        return self.pdf(u, v)

    def d_vfun_d_v(self, u, v):
        """
        Evaluate the derivative of the copula vfun w.r.t. v.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        return self.pdf(u, v)

    def d_inv_hfun_d_u(self, u, v):
        """
        Evaluate the derivative of the inverse of the copula hfun w.r.t. u.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        return 1. / self.pdf(self.inv_hfun(u, v), v)

    def d_inv_vfun_d_v(self, u, v):
        """
        Evaluate the derivative of the inverse of the copula vfun w.r.t. v.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        return 1. / self.pdf(u, self.inv_vfun(u, v))

    def d_inv_hfun_d_v(self, u, v):
        """
        Evaluate the derivative of the inverse of the copula hfun w.r.t. v.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_v(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_u(self, u, v):
        """
        Evaluate the derivative of the inverse of the copula vfun w.r.t. u.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_u(u, xx) / self.pdf(u, xx)

    def d_inv_hfun_d_par(self, u, v):
        """
        Evaluate the derivative of the inverse of the copula hfun w.r.t. the parameter.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        xx = self.inv_hfun(u, v)
        return -1. * self.d_hfun_d_par(xx, v) / self.pdf(xx, v)

    def d_inv_vfun_d_par(self, u, v):
        """
        Evaluate the derivative of the inverse of the copula vfun w.r.t. the parameter.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of observations for the first variable.
        v : :class:`numpy.ndarray`
            Array of observations for the second variable.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        xx = self.inv_vfun(u, v)
        return -1. * self.d_vfun_d_par(u, xx) / self.pdf(u, xx)

    def sim(self, n_obs=100):
        """
        Simulate random observations from the copula.

        Parameters
        ----------
        n_obs :
            The number of observations to simulate.
        Returns
        -------
        res : :class:`numpy.ndarray`
        """
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_hfun(u[:, 0], u[:, 1])
        return u


class ClaytonCopula(Copula):
    """Clayton copula (bivariate).

    Parameters
    ----------
    par : float
        Parameter of the Clayton copula.
        Default is np.nan.

    rotation: int
        Rotation (0, 90, 180 or 270)
        Default is 0.

    Examples
    --------
    # ToDo: add an example here

    Notes
    -----
    The cdf of the Clayton copula with parameter :math:`\\theta \\in (0, \\infty)` is given by

    .. math::
        C(u, v; \\theta) = (u^{-\\theta} + v^{-\\theta} - 1)^{-\\frac{1}{\\theta}}.

    """
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
    """Frank copula (bivariate).

    Parameters
    ----------
    par : float
        Parameter of the Frank copula.
        Default is np.nan.

    Examples
    --------
    # ToDo: add an example here

    Notes
    -----
    The cdf of the Frank copula with parameter :math:`\\theta \\in (-\\infty, \\infty) \\setminus \\lbrace 0 \\rbrace`
    is given by

    .. math::
        C(u, v; \\theta) = -\\frac{1}{\\theta} \\log\\bigg(\\frac{1}{1-\\exp(-\\theta)}
        \\big[(1-\\exp(-\\theta)) - (1-\\exp(-\\theta u)) (1-\\exp(-\\theta v)) \\big] \\bigg).

    """
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
    """Gaussian copula (bivariate).

    Parameters
    ----------
    par : float
        Parameter of the Gaussian copula.
        Default is np.nan.

    Examples
    --------
    # ToDo: add an example here

    Notes
    -----
    The pdf of the Gaussian copula with parameter :math:`\\theta \\in (-1, 1)` is given by

    .. math::
        c(u, v; \\theta) = \\frac{1}{\\sqrt{1-\\theta^2}} \\exp\\bigg(-
        \\frac{\\theta^2 (x^2 + y^2) - 2 \\theta x y}{2 (1-\\theta^2)}\\bigg),

    with :math:`x = \\Phi^{-1}(u)` and :math:`y = \\Phi^{-1}(v)`.
    """
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
    """Gumbel copula (bivariate).

    Parameters
    ----------
    par : float
        Parameter of the Gumbel copula.
        Default is np.nan.

    rotation: int
        Rotation (0, 90, 180 or 270)
        Default is 0.

    Examples
    --------
    # ToDo: add an example here

    Notes
    -----
    The cdf of the Gumbel copula with parameter :math:`\\theta \\in (1, \\infty)` is given by

    .. math::
        C(u, v; \\theta) = \\exp\\big(-\\big[ (-\\log(u))^\\theta + (-\\log(v))^\\theta \\big]^{\\frac{1}{\\theta}} \\big).

    """
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

    def inv_hfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        f_a = self._hfun(par=self.par, u=1e-12, v=v) - u
        f_b = self._hfun(par=self.par, u=1-1e-12, v=v) - u
        if (f_a * f_b < 0.).all():
            res = super(GumbelCopula, self).inv_hfun(u, v)
        else:
            ind_sign_change = (f_a * f_b < 0.)
            no_sign_change = np.logical_not(ind_sign_change)
            res = np.full_like(u, np.nan)
            res[ind_sign_change] = super(GumbelCopula, self).inv_hfun(u[ind_sign_change], v[ind_sign_change])

            xx = np.full(no_sign_change.sum(), np.nan)
            xx[np.abs(f_a[no_sign_change]) < np.abs(f_b[no_sign_change])] = 1e-12
            xx[np.abs(f_b[no_sign_change]) < np.abs(f_a[no_sign_change])] = 1-1e-12
            res[no_sign_change] = xx

            xx = self._hfun(par=self.par, u=res[no_sign_change], v=v[no_sign_change]) - u[no_sign_change]
            if np.abs(xx).min() > 1e-6:
                raise ValueError(f'inv_hfun: Root search failed')
        return self._trim_obs(res)

    def inv_vfun(self, u, v):
        u = self._trim_obs(u)
        v = self._trim_obs(v)
        f_a = self._vfun(par=self.par, u=u, v=1e-12) - v
        f_b = self._vfun(par=self.par, u=u, v=1-1e-12) - v
        if (f_a * f_b < 0.).all():
            res = super(GumbelCopula, self).inv_vfun(u, v)
        else:
            ind_sign_change = (f_a * f_b < 0.)
            no_sign_change = np.logical_not(ind_sign_change)
            res = np.full_like(u, np.nan)
            res[ind_sign_change] = super(GumbelCopula, self).inv_vfun(u[ind_sign_change], v[ind_sign_change])

            xx = np.full(no_sign_change.sum(), np.nan)
            xx[np.abs(f_a[no_sign_change]) < np.abs(f_b[no_sign_change])] = 1e-12
            xx[np.abs(f_b[no_sign_change]) < np.abs(f_a[no_sign_change])] = 1-1e-12
            res[no_sign_change] = xx

            xx = self._vfun(par=self.par, u=u[no_sign_change], v=res[no_sign_change]) - v[no_sign_change]
            if np.abs(xx).min() > 1e-6:
                raise ValueError(f'inv_vfun: Root search failed')
        return self._trim_obs(res)


class IndepCopula(Copula):
    """Independence / product copula (bivariate).

    Examples
    --------
    # ToDo: add an example here

    Notes
    -----
    The cdf of the independence / product copula is given by

    .. math::
        C(u, v) = u v.

    """
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
