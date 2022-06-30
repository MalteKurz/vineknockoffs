import numpy as np
from scipy.stats import norm

from .vine_copulas import DVineCopula
from ._utils_gaussian_knockoffs import sdp_solver, ecorr_solver
from ._utils_vine_copulas import dvine_pcorr
from .copulas import GaussianCopula
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