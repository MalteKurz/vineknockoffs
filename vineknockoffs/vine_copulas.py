import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

from .copulas import cop_select, IndepCopula
from ._utils_gaussian_knockoffs import sdp_solver, ecorr_solver
from ._utils_vine_copulas import dvine_pcorr
from .copulas import GaussianCopula


class KDE(sm.nonparametric.KDEUnivariate):
    def ppf(self, x):
        return self.i_cdf(x)


class DVineCopula:

    def __init__(self, copulas):
        self._copulas = copulas

    @property
    def copulas(self):
        return self._copulas

    @property
    def n_vars(self):
        return len(self.copulas) + 1

    def sim(self, n_obs=100, w=None):
        if w is None:
            w = np.random.uniform(size=(n_obs, self.n_vars))

        a = np.full_like(w, np.nan)
        b = np.full_like(w, np.nan)
        u = np.full_like(w, np.nan)

        u[:, 0] = w[:, 0]
        a[:, 0] = w[:, 0]
        b[:, 0] = w[:, 0]

        # for i in np.arange(1, self.n_vars):
        #     a[:, 0] = w[:, i]
        #     for j in np.arange(0, i-1):
        #         tree = i-j
        #         cop = j
        #         a[:, j+1] = self.copulas[tree][cop].vfun(b[:, j], a[:, j])
        #     u[:, i] = a[:, i]
        #     b[:, i] = a[:, i]
        #     for j in np.arange(i-2, -1, -1):
        #         tree = i-j
        #         cop = j
        #         b[:, j] = self.copulas[tree][cop].hfun(b[:, j], a[:, j+1])

        for i in np.arange(2, self.n_vars+1):
            a[:, 0] = w[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = j
                cop = i-j
                a[:, i-j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, i-j-1], a[:, i-j-1])
            u[:, i-1] = a[:, i-1]
            if i < self.n_vars:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = j
                    cop = i-j
                    b[:, i-j-1] = self.copulas[tree-1][cop-1].hfun(b[:, i-j-1], a[:, i-j])
        return u

    def compute_pits(self, u):
        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        w = np.full_like(u, np.nan)

        w[:, 0] = u[:, 0]
        a[:, 0] = u[:, 0]
        b[:, 0] = u[:, 0]

        for i in np.arange(2, self.n_vars+1):
            a[:, i-1] = u[:, i-1]
            for j in np.arange(1, i):
                tree = j
                cop = i-j
                a[:, i-j-1] = self.copulas[tree-1][cop-1].vfun(b[:, i-j-1], a[:, i-j])
            w[:, i-1] = a[:, 0]
            if i < self.n_vars:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = j
                    cop = i-j
                    b[:, i-j-1] = self.copulas[tree-1][cop-1].hfun(b[:, i-j-1], a[:, i-j])
        return w

    @classmethod
    def cop_select(cls, u, families='all', indep_test=True):
        n_vars = u.shape[1]
        copulas = [[IndepCopula()] * j for j in np.arange(n_vars - 1, 0, -1)]

        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        xx = None

        for i in np.arange(1, n_vars):
            a[:, i] = u[:, i]
            b[:, i-1] = u[:, i-1]

        for j in np.arange(1, n_vars):
            tree = j
            for i in np.arange(1, n_vars-j+1):
                cop = i
                copulas[tree-1][cop-1] = cop_select(b[:, i-1], a[:, i+j-1],
                                                    families=families, indep_test=indep_test)
                if i < n_vars-j:
                    xx = copulas[tree-1][cop-1].hfun(b[:, i-1], a[:, i+j-1])
                if i > 1:
                    a[:, i+j-1] = copulas[tree-1][cop-1].vfun(b[:, i-1], a[:, i+j-1])
                if i < n_vars-j:
                    b[:, i-1] = xx

        return cls(copulas)


class VineKnockoffs:

    def __init__(self, dvine=None, marginals=None):
        if (dvine is not None) & (not isinstance(dvine, DVineCopula)):
            raise TypeError('dvine must be of DVineCopula type. '
                            f'{str(dvine)} of type {str(type(dvine))} was passed.')
        self._dvine = dvine
        self._marginals = marginals

    def generate(self, x_test):
        n_obs = x_test.shape[0]
        n_vars = x_test.shape[1]
        u_test = np.full_like(x_test, np.nan)
        for i_var in range(n_vars):
            u_test[:, i_var] = self._marginals[i_var].cdf(x_test[:, i_var])

        sub_dvine = DVineCopula([self._dvine.copulas[tree][:n_vars-tree-1] for tree in np.arange(0, n_vars-1)])
        u_pits = sub_dvine.compute_pits(u_test)
        knockoff_pits = np.random.uniform(size=(n_obs, n_vars))

        u_sim = self._dvine.sim(w=np.hstack((u_pits, knockoff_pits)))
        u_knockoffs = u_sim[:, n_vars:]

        x_knockoffs = np.full_like(x_test, np.nan)
        for i_var in range(n_vars):
            x_knockoffs[:, i_var] = self._marginals[i_var].ppf(u_knockoffs[:, i_var])
        return x_knockoffs

    # ToDo Implement Gaussian copula knockoffs
    def fit_gaussian_knockoffs(self, x_train, algo='sdp'):
        n_vars = x_train.shape[1]
        mus = np.mean(x_train, axis=0)
        sigmas = np.std(x_train, axis=0)
        self._marginals = [norm(loc=mus[i_var], scale=sigmas[i_var])
                           for i_var in range(n_vars)]

        corr_mat = np.corrcoef(x_train.transpose())

        if algo == 'sdp':
            s_vec = sdp_solver(corr_mat)
        else:
            assert algo == 'ecorr'
            s_vec = ecorr_solver(corr_mat)

        g_mat = np.vstack((np.hstack((corr_mat, corr_mat - np.diag(s_vec))),
                           np.hstack((corr_mat - np.diag(s_vec), corr_mat))))
        pcorrs = dvine_pcorr(g_mat)
        copulas = [[GaussianCopula(rho) for rho in xx] for xx in pcorrs]
        self._dvine = DVineCopula(copulas)

        return self

