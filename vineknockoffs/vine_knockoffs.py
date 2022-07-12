from copy import deepcopy
import numpy as np
from scipy.stats import norm, bernoulli

from .copulas import cop_select, GaussianCopula, IndepCopula, FrankCopula
from .knockoffs import KockoffsLoss
from .vine_copulas import DVineCopula

from ._utils_gaussian_knockoffs import sdp_solver, ecorr_solver
from ._utils_kde import KDEMultivariateWithInvCdf, KDE1D
from ._utils_vine_copulas import dvine_pcorr, d_vine_structure_select


class VineKnockoffs:

    def __init__(self, dvine=None, marginals=None):
        if (dvine is not None) & (not isinstance(dvine, DVineCopula)):
            raise TypeError('dvine must be of DVineCopula type. '
                            f'{str(dvine)} of type {str(type(dvine))} was passed.')
        self._dvine = dvine
        self._marginals = marginals
        if dvine is not None:
            self._dvine_structure = np.arange(self._dvine.n_vars)
            self._inv_dvine_structure = np.argsort(self._dvine_structure)
        else:
            self._dvine_structure = np.array([])
            self._inv_dvine_structure = np.array([])

    @property
    def n_pars_upper_trees(self):
        n_vars = int(self._dvine.n_vars/2)
        from_tree = n_vars - 1
        n_pars_upper_trees = np.sum([np.sum([cop.n_pars for cop in tree]) for tree in self._dvine.copulas[from_tree:]])
        return n_pars_upper_trees

    @property
    def dvine_structure(self):
        return self._dvine_structure

    @dvine_structure.setter
    def dvine_structure(self, value):
        self._dvine_structure = value
        self._inv_dvine_structure = np.argsort(self._dvine_structure)

    @property
    def inv_dvine_structure(self):
        return self._inv_dvine_structure

    def generate(self, x_test, knockoff_eps=None):
        n_obs = x_test.shape[0]
        n_vars = x_test.shape[1]
        u_test = np.full_like(x_test, np.nan)
        for i_var in range(n_vars):
            u_test[:, i_var] = self._marginals[i_var].cdf(x_test[:, i_var])

        # apply dvine structure / variable order
        u_test = u_test[:, self.dvine_structure]

        sub_dvine = DVineCopula([self._dvine.copulas[tree][:n_vars-tree-1] for tree in np.arange(0, n_vars-1)])
        u_pits = sub_dvine.compute_pits(u_test)
        if knockoff_eps is None:
            knockoff_pits = np.random.uniform(size=(n_obs, n_vars))
        else:
            knockoff_pits = knockoff_eps

        u_sim = self._dvine.sim(w=np.hstack((u_pits, knockoff_pits)))
        u_knockoffs = u_sim[:, n_vars:]

        # get back order of variables
        u_knockoffs = u_knockoffs[:, self.inv_dvine_structure]

        x_knockoffs = np.full_like(x_test, np.nan)
        for i_var in range(n_vars):
            x_knockoffs[:, i_var] = self._marginals[i_var].ppf(u_knockoffs[:, i_var])
        return x_knockoffs

    def generate_par_jacobian(self, x_test, knockoff_eps=None, which_par='upper only',
                              return_x_knockoffs=False):
        n_obs = x_test.shape[0]
        n_vars = x_test.shape[1]
        if knockoff_eps is None:
            knockoff_pits = np.random.uniform(size=(n_obs, n_vars))
        else:
            knockoff_pits = knockoff_eps

        u_test = np.full_like(x_test, np.nan)
        for i_var in range(n_vars):
            u_test[:, i_var] = self._marginals[i_var].cdf(x_test[:, i_var])

        # apply dvine structure / variable order
        u_test = u_test[:, self.dvine_structure]

        sub_dvine = DVineCopula([self._dvine.copulas[tree][:n_vars-tree-1] for tree in np.arange(0, n_vars-1)])

        if which_par == 'upper only':
            n_pars = self.n_pars_upper_trees
            u_pits = sub_dvine.compute_pits(u_test)
            u_sim, u_sim_jacobian = self._dvine.sim_par_jacobian_fast(w=np.hstack((u_pits, knockoff_pits)),
                                                                      from_tree=n_vars, return_u=True)
        else:
            assert which_par == 'all'
            n_pars = self._dvine.n_pars
            u_pits = sub_dvine.compute_pits(u_test)
            u_pits_d_par = sub_dvine.compute_pits_par_jacobian(u_test)
            w_jacobian = np.zeros((n_obs, self._dvine.n_vars, self._dvine.n_pars))
            ind_par = 0
            ind_par_sub_dvine = 0
            for tree in np.arange(1, n_vars):
                for cop in np.arange(1, self._dvine.n_vars - tree + 1):
                    cop_n_pars = self._dvine.copulas[tree - 1][cop - 1].n_pars
                    if cop_n_pars > 0:
                        assert self._dvine.copulas[tree - 1][cop - 1].n_pars == 1
                        if cop < n_vars - tree + 1:
                            w_jacobian[:, :n_vars, ind_par] = u_pits_d_par[:, :, ind_par_sub_dvine]
                            ind_par_sub_dvine += self._dvine.copulas[tree - 1][cop - 1].n_pars
                        ind_par += self._dvine.copulas[tree - 1][cop - 1].n_pars

            u_sim, u_sim_jacobian = self._dvine.sim_par_jacobian_fast(w=np.hstack((u_pits, knockoff_pits)),
                                                                      w_jacobian=w_jacobian, return_u=True)

        u_knockoffs = u_sim[:, n_vars:]
        u_knockoffs_jacobian = u_sim_jacobian[:, n_vars:, :]

        # get back order of variables
        u_knockoffs = u_knockoffs[:, self.inv_dvine_structure]
        u_knockoffs_jacobian = u_knockoffs_jacobian[:, self.inv_dvine_structure, :]

        x_knockoffs = np.full_like(x_test, np.nan)
        d_x_d_u = np.full_like(x_test, np.nan)
        for i_var in range(n_vars):
            x_knockoffs[:, i_var] = self._marginals[i_var].ppf(u_knockoffs[:, i_var])
            d_x_d_u[:, i_var] = 1 / self._marginals[i_var].pdf(x_knockoffs[:, i_var])

        x_knockoffs_jacobian = np.full_like(u_knockoffs_jacobian, np.nan)
        for i_par in range(n_pars):
            x_knockoffs_jacobian[:, :, i_par] = d_x_d_u * u_knockoffs_jacobian[:, :, i_par]

        if return_x_knockoffs:
            return x_knockoffs, x_knockoffs_jacobian
        else:
            return x_knockoffs_jacobian

    def fit_vine_copula_knockoffs(self, x_train,
                                  marginals='kde1d',
                                  families='all', rotations=True, indep_test=True,
                                  upper_tree_cop_fam_heuristic='lower tree families',
                                  sgd=True, sgd_lr=0.01, sgd_gamma=0.9, sgd_n_batches=5, sgd_n_iter=20,
                                  sgd_which_par='all',
                                  loss_alpha=1., loss_delta_sdp_corr=1., loss_gamma=1., loss_delta_corr=0.):

        # fit gaussian copula knockoffs (marginals are fitted and parameters for the decorrelation tree are determined)
        self.fit_gaussian_copula_knockoffs(x_train, marginals=marginals, algo='sdp')

        # select copula families via AIC / MLE
        n_vars = x_train.shape[1]
        n_vars_x2 = n_vars * 2
        u_train = np.full_like(x_train, np.nan)
        for i_var in range(n_vars):
            u_train[:, i_var] = self._marginals[i_var].cdf(x_train[:, i_var])

        # determine dvine structure / variable order
        self.dvine_structure = d_vine_structure_select(u_train)
        u_train = u_train[:, self.dvine_structure]

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
                                                                            families=families, rotations=rotations,
                                                                            indep_test=indep_test)
                    else:
                        self._dvine.copulas[tree - 1][cop - 1] = deepcopy(
                            self._dvine.copulas[tree - 1][cop - 1 - n_vars])
                elif tree == n_vars:
                    # decorrelation tree (do nothing; take Gaussian copula determined via fit_gaussian_copula_knockoffs)
                    assert isinstance(self._dvine.copulas[tree - 1][cop - 1], GaussianCopula)
                else:
                    assert tree > n_vars
                    # upper trees
                    if upper_tree_cop_fam_heuristic == 'Gaussian':
                        assert isinstance(self._dvine.copulas[tree - 1][cop - 1], GaussianCopula)
                    else:
                        assert upper_tree_cop_fam_heuristic == 'lower tree families'
                        assert isinstance(self._dvine.copulas[tree - 1][cop - 1], GaussianCopula)
                        lower_tree_cop = self._dvine.copulas[tree - n_vars - 1][cop - 1]
                        par_gaussian_ko = self._dvine.copulas[tree - 1][cop - 1].par
                        tau_gaussian_ko = GaussianCopula().par2tau(par_gaussian_ko)
                        if isinstance(lower_tree_cop, IndepCopula):
                            this_copula = GaussianCopula(par_gaussian_ko)
                        elif (np.abs(tau_gaussian_ko) > 0.9) & isinstance(lower_tree_cop, FrankCopula):
                            this_copula = GaussianCopula(par_gaussian_ko)
                        elif (tau_gaussian_ko < 0.) & (lower_tree_cop.rotation in [0, 180]):
                            this_copula = GaussianCopula(par_gaussian_ko)
                        elif (tau_gaussian_ko > 0.) & (lower_tree_cop.rotation in [90, 270]):
                            this_copula = GaussianCopula(par_gaussian_ko)
                        else:
                            this_copula = deepcopy(lower_tree_cop)
                            this_copula.set_par_w_bound_check(this_copula.tau2par(tau_gaussian_ko))

                        self._dvine.copulas[tree - 1][cop - 1] = this_copula

                if i < n_vars_x2 - j:
                    xx = self._dvine.copulas[tree - 1][cop - 1].hfun(b[:, i - 1], a[:, i + j - 1])
                if i > 1:
                    a[:, i + j - 1] = self._dvine.copulas[tree - 1][cop - 1].vfun(b[:, i - 1], a[:, i + j - 1])
                if i < n_vars_x2 - j:
                    b[:, i - 1] = xx

        if sgd:
            self.fit_sgd(x_train=x_train,
                         lr=sgd_lr, gamma=sgd_gamma, n_batches=sgd_n_batches, n_iter=sgd_n_iter,
                         which_par=sgd_which_par,
                         loss_alpha=loss_alpha, loss_delta_sdp_corr=loss_delta_sdp_corr,
                         loss_gamma=loss_gamma, loss_delta_corr=loss_delta_corr)

        return self

    def fit_sgd(self, x_train,
                lr=0.01, gamma=0.9, n_batches=5, n_iter=20,
                which_par='all',
                loss_alpha=1., loss_delta_sdp_corr=1., loss_gamma=1., loss_delta_corr=0.):
        n_obs = x_train.shape[0]
        n_vars = x_train.shape[1]
        loss_obj = KockoffsLoss(alpha=loss_alpha, delta_sdp_corr=loss_delta_sdp_corr,
                                gamma=loss_gamma, delta_corr=loss_delta_corr)
        if which_par == 'all':
            start_tree = 1
        else:
            assert which_par == 'upper only'
            start_tree = n_vars
        par_vec = self._dvine.get_par_vec(from_tree=start_tree)

        losses = np.full(n_iter, np.nan)
        x_knockoffs = self.generate(x_test=x_train)
        swap_inds = np.arange(0, n_vars)[bernoulli.rvs(0.5, size=n_vars) == 1]
        loss_vals = loss_obj.eval(x=x_train, x_knockoffs=x_knockoffs,
                                  swap_inds=swap_inds, sdp_corr=self._sdp_corr)
        print(loss_vals)
        losses[0] = loss_vals[0]
        n_obs_batch = int(np.floor(n_obs / n_batches))
        for i_iter in np.arange(1, n_iter):
            ind_shuffled = np.arange(n_obs)
            np.random.shuffle(ind_shuffled)
            x_data = x_train.copy()[ind_shuffled, :]
            update_step = 0.
            for i_batch in range(n_batches):
                ind_start = i_batch * n_obs_batch
                if i_batch < n_batches - 1:
                    ind_end = (i_batch + 1) * n_obs_batch
                else:
                    ind_end = n_obs

                # generate knockoffs
                knockoff_eps = np.random.uniform(size=(n_obs, n_vars))

                x_knockoffs[ind_start:ind_end, :] = self.generate(x_test=x_data[ind_start:ind_end, :],
                                                                  knockoff_eps=knockoff_eps[ind_start:ind_end, :])
                x_knockoffs_deriv = self.generate_par_jacobian(x_test=x_data[ind_start:ind_end, :],
                                                               knockoff_eps=knockoff_eps[ind_start:ind_end, :],
                                                               which_par=which_par)

                swap_inds = np.arange(0, n_vars)[bernoulli.rvs(0.5, size=n_vars) == 1]
                loss_grad = loss_obj.deriv(x=x_data[ind_start:ind_end, :],
                                           x_knockoffs=x_knockoffs[ind_start:ind_end, :],
                                           x_knockoffs_deriv=x_knockoffs_deriv,
                                           swap_inds=swap_inds, sdp_corr=self._sdp_corr)[0]

                update_step = gamma * update_step + lr * loss_grad
                par_vec = par_vec - update_step

                self._dvine.set_par_vec(par_vec=par_vec, from_tree=start_tree, assert_to_bounds=True)

            swap_inds = np.arange(0, n_vars)[bernoulli.rvs(0.5, size=n_vars) == 1]
            loss_vals = loss_obj.eval(x=x_data, x_knockoffs=x_knockoffs,
                                      swap_inds=swap_inds, sdp_corr=self._sdp_corr)
            losses[i_iter] = loss_vals[0]
            print(i_iter)
            print(loss_vals)
        return self

    def fit_gaussian_copula_knockoffs(self, x_train, marginals='kde1d', algo='sdp'):
        # ToDo May add alternative methods for the marginals (like parameteric distributions)
        n_vars = x_train.shape[1]

        if marginals == 'kde1d':
            self._marginals = [KDE1D().fit(x_train[:, i_var])
                               for i_var in range(n_vars)]
        else:
            assert marginals == 'kde_statsmodels'
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
        copulas = [[GaussianCopula().set_par_w_bound_check(rho) for rho in xx] for xx in pcorrs]
        self._dvine = DVineCopula(copulas)
        self.dvine_structure = np.arange(int(self._dvine.n_vars/2))
        self._sdp_corr = 1. - s_vec

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
        self.dvine_structure = np.arange(int(self._dvine.n_vars/2))

        return self
