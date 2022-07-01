import numpy as np
from scipy.stats import norm

from .vine_copulas import DVineCopula
from ._utils_gaussian_knockoffs import sdp_solver, ecorr_solver
from ._utils_vine_copulas import dvine_pcorr
from .copulas import cop_select, GaussianCopula
from ._utils_kde import KDEMultivariateWithInvCdf


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

    def fit_vine_copula_knockoffs(self, x_train, families='all', indep_test=True):
        # fit gaussian copula knockoffs (marginals are fitted and parameters for the decorrelation tree are determined)
        self.fit_gaussian_copula_knockoffs(x_train, algo='sdp')

        # select copula families via AIC / MLE
        n_vars = x_train.shape[1]
        n_vars_x2 = n_vars * 2
        u_train = np.full_like(x_train, np.nan)
        for i_var in range(n_vars):
            u_train[:, i_var] = self._marginals[i_var].cdf(x_train[:, i_var])
        uu = np.hstack((u_train, u_train))

        a = np.full_like(uu, np.nan)
        b = np.full_like(uu, np.nan)
        xx = None

        for i in np.arange(1, n_vars_x2):
            a[:, i] = uu[:, i]
            b[:, i - 1] = uu[:, i - 1]

        for j in np.arange(1, n_vars_x2):
            tree = j
            for i in np.arange(1, n_vars_x2 - j + 1):
                cop = i
                if tree < n_vars:
                    # lower trees
                    if cop <= n_vars:
                        self._dvine.copulas[tree - 1][cop - 1] = cop_select(b[:, i - 1], a[:, i + j - 1],
                                                                            families=families, indep_test=indep_test)
                    else:
                        self._dvine.copulas[tree - 1][cop - 1] = self._dvine.copulas[tree - 1][cop - 1 - n_vars]
                elif tree == n_vars:
                    # decorrelation tree (do nothing; take Gaussian copula determined via fit_gaussian_copula_knockoffs)
                    assert isinstance(self._dvine.copulas[tree - 1][cop - 1], GaussianCopula)
                else:
                    assert tree > n_vars
                    # upper trees (standard selection but deactivate the independence test)
                    self._dvine.copulas[tree - 1][cop - 1] = cop_select(b[:, i - 1], a[:, i + j - 1],
                                                                        families=families, indep_test=False)

                if i < n_vars_x2 - j:
                    xx = self._dvine.copulas[tree - 1][cop - 1].hfun(b[:, i - 1], a[:, i + j - 1])
                if i > 1:
                    a[:, i + j - 1] = self._dvine.copulas[tree - 1][cop - 1].vfun(b[:, i - 1], a[:, i + j - 1])
                if i < n_vars_x2 - j:
                    b[:, i - 1] = xx

        return self

    def fit_gaussian_copula_knockoffs(self, x_train, algo='sdp'):
        # ToDo May add alternative methods for the marginals (like parameteric distributions)
        n_vars = x_train.shape[1]
        self._marginals = [KDEMultivariateWithInvCdf(x_train[:, i_var], 'c')
                           for i_var in range(n_vars)]
        u_train = np.full_like(x_train, np.nan)
        x_gaussian_train = np.full_like(x_train, np.nan)
        for i_var in range(n_vars):
            u_train[:, i_var] = self._marginals[i_var].cdf(x_train[:, i_var])
            x_gaussian_train[:, i_var] = norm.ppf(u_train[:, i_var])

        corr_mat = np.corrcoef(x_gaussian_train.transpose())

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