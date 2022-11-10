from copy import deepcopy
import numpy as np
from scipy.stats import norm, bernoulli

from .copulas import cop_select, GaussianCopula, IndepCopula, FrankCopula
from .knockoffs import KnockoffsLoss
from .vine_copulas import DVineCopula

from ._utils_gaussian_knockoffs import sdp_solver, ecorr_solver
from ._utils_kde import KDEMultivariateWithInvCdf
from ._utils_vine_copulas import dvine_pcorr, d_vine_structure_select
try:
    from ._utils_kde1d import KDE1D
except ImportError:
    _has_kde1d = False
else:
    _has_kde1d = True


class VineKnockoffs:
    """Vine copula knockoffs.

    Parameters
    ----------
    dvine : None or :class:`DVineCopula` object
        The :class:`DVineCopula` object specifies the vine copula model used to generate knockoffs. If it is set to None
        the vine copula knockoff model can be learned from data with the methods ``fit_vine_copula_knockoffs()``,
        ``fit_gaussian_copula_knockoffs()`` or ``fit_gaussian_knockoffs()``.
        Default is ``None``.

    marginals: list
        The marginal distributions for the vine copula knockoff model. Must be a list of `n_vars` distributions which
        implement the methods ``cdf()`` and ``ppf()``. If it is set to None the marginal distributions can be estimated
        from data with the methods ``fit_marginals()``, which is also called by ``fit_vine_copula_knockoffs()``,
        ``fit_gaussian_copula_knockoffs()`` or ``fit_gaussian_knockoffs()``.
        Default is ``None``.

    dvine_structure: :class:`numpy.array`
        The D-vine structure (order of variables) for the vine copula knockoff model.
        Default is ``None``.

    Examples
    --------
    # ToDo: add an example here
    """

    def __init__(self, dvine=None, marginals=None, dvine_structure=None):
        if (dvine is not None) & (not isinstance(dvine, DVineCopula)):
            raise TypeError('dvine must be of DVineCopula type. '
                            f'{str(dvine)} of type {str(type(dvine))} was passed.')
        # ToDo: add some more exception handling for the parameters
        self._dvine = dvine
        self._marginals = marginals
        if dvine is not None:
            if dvine_structure is not None:
                self._dvine_structure = dvine_structure
                self._inv_dvine_structure = np.argsort(self._dvine_structure)
            else:
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
            u_pits, u_pits_d_par = sub_dvine.compute_pits_par_jacobian_fast(u_test, return_w=True)
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
                                  vine_structure='select_tsp_r',  # 'select_tsp_r', 'select_tsp_py', '1:n'
                                  upper_tree_cop_fam_heuristic='Gaussian',
                                  sgd=False, sgd_lr=0.01, sgd_gamma=0.9, sgd_n_batches=5, sgd_n_iter=20,
                                  sgd_which_par='all',
                                  loss_alpha=1., loss_delta_sdp_corr=1., loss_gamma=1., loss_delta_corr=0.,
                                  gau_cop_algo='sdp'):
        """
        Estimate a vine copula knockoff model.

        Parameters
        ----------
        x_train : :class:`numpy.ndarray`
            Array of covariates.

        marginals : str
            A str (``'kde1d'`` or ``'kde_statsmodels'``) specifying the estimator for the marginal distributions.
            ``'kde1d'``: The univariate kernel density estimator implemented in the R package kde1d (via rpy2).
            ``'kde_statsmodels'``: The univariate version of the kernel density estimator ``KDEMultivariate``
            implemented in the Python package statsmodels.
            Default is ``'kde1d'``.

        families : str
            The set of possible copula families explored via minimizing the AIC. Currently, the only implemented choice
            is ``'all'`` (i.e., Clayton, Frank, Gaussian, and Gumbel).
            Default is ``'all'``.

        rotations : bool
            Indicates whether rotated versions of the Clayton and Gumbel copula should be considered.
            Default is ``True``.

        indep_test : bool
            Indicates whether an independence test should be performed before choosing a copula family with the AIC. If
            independence cannot be rejected, an Independence copula is being assigned to the corresponding vine edge.
            Default is ``True``.

        vine_structure : str
            A str (``'select_tsp_r'``, ``'select_tsp_py'`` or ``'1:n'``) specifying how the structure of the D-vine is
            selected.
            ``'select_tsp_r'``: Maximize the dependence in the first tree by solving a TSP with the R package TSP.
            ``'select_tsp_py'``: Maximize the dependence in the first tree by solving a TSP with the Python package
            python_tsp.
            ``'1:n'``: Use the natural order of the variables X_1, X_2, ..., X_d-1, X_d.
            Default is ``'select_tsp_r'``.

        upper_tree_cop_fam_heuristic : str
            A str (``'Gaussian'`` or ``'lower tree families'``) specifying the heuristic used to select the copula
            families in the higher trees (from the (d+1)-th tree on).
            ``'Gaussian'``: Gaussian copulas corresponding to the partial correlation vine are assigned to the edges in
            the higher trees.
            ``'lower tree families'``: The copula families from the lower trees are mirrored to the higher trees.
            Default is ``'Gaussian'``.

        sgd : bool
            Indicates whether the parameters of the vine copula knockoff model should be tuned using a stochastic
            gradient descent algorithm.
            Default is ``False``.

        sgd_lr : float
            The learning rate for the SGD algorithm.
            Default is ``0.01``.

        sgd_gamma : float
            SGD momentum parameter.
            Default is ``0.9``.

        sgd_n_batches : int
            Number of batches in the SGD algorithm.
            Default is ``5``.

        sgd_n_iter : int
            Maximum number of iterations of the SGD algorithm.
            Default is ``20``.

        sgd_which_par : str
            A str (``''all''`` or ``''upper only''``) specifying whether all parameters or only the parameters of the
            upper trees should be optimized with the SGD algorithm.
            Default is ``''all''``.

        loss_alpha : float
            Parameter alpha in the SGD loss function.
            Default is ``1.``.

        loss_delta_sdp_corr : float
            Parameter delta_sdp_corr in the SGD loss function.
            Default is ``1.``.

        loss_gamma : float
            Parameter gamma in the SGD loss function.
            Default is ``1.``.

        loss_delta_corr : float
            Parameter delta_corr in the SGD loss function.
            Default is ``0.``.

        gau_cop_algo : str
            A str (``'sdp'`` or ``''ecorr''``) specifying the algorithm used (semidefinite program or equicorrelation)
            to obtain the parameters (the partial correlation vine) of the Gaussian copula knockoffs.
        """

        if (not isinstance(marginals, str)) | (marginals not in ['kde1d', 'kde_statsmodels']):
            raise ValueError('marginals must be "kde1d" or "kde_statsmodels". '
                             f'Got {str(marginals)}.')

        if (not isinstance(families, str)) | (families not in ['all']):
            raise ValueError('families must be "all". '
                             f'Got {str(families)}.')

        if not isinstance(rotations, bool):
            raise TypeError('rotations must be True or False. '
                            f'Got {str(rotations)}.')

        if not isinstance(indep_test, bool):
            raise TypeError('indep_test must be True or False. '
                            f'Got {str(indep_test)}.')

        if (not isinstance(vine_structure, str)) | (vine_structure not in ['select_tsp_r', 'select_tsp_py', '1:n']):
            raise ValueError('vine_structure must be "select_tsp_r", "select_tsp_py" or "1:n". '
                             f'Got {str(vine_structure)}.')

        if (not isinstance(upper_tree_cop_fam_heuristic, str)) | (upper_tree_cop_fam_heuristic not in
                                                                  ['Gaussian', 'lower tree families']):
            raise ValueError('upper_tree_cop_fam_heuristic must be "Gaussian" or "lower tree families". '
                             f'Got {str(upper_tree_cop_fam_heuristic)}.')

        if not isinstance(sgd, bool):
            raise TypeError('sgd must be True or False. '
                            f'Got {str(sgd)}.')

        if isinstance(sgd_lr, int):
            sgd_lr = float(sgd_lr)
        if not isinstance(sgd_lr, float):
            raise TypeError('sgd_lr must be of float type. '
                            f'{str(sgd_lr)} of type {str(type(sgd_lr))} was passed.')
        # ToDo: Allowed values for the learning rate?

        if isinstance(sgd_gamma, int):
            sgd_gamma = float(sgd_gamma)
        if not isinstance(sgd_gamma, float):
            raise TypeError('sgd_gamma must be of float type. '
                            f'{str(sgd_gamma)} of type {str(type(sgd_gamma))} was passed.')
        # ToDo: Allowed values for gamma / momentum?

        if not isinstance(sgd_n_batches, int):
            raise TypeError('sgd_n_batches must be of int type. '
                            f'{str(sgd_n_batches)} of type {str(type(sgd_n_batches))} was passed.')
        # ToDo: Positive integers?

        if not isinstance(sgd_n_iter, int):
            raise TypeError('sgd_n_iter must be of int type. '
                            f'{str(sgd_n_iter)} of type {str(type(sgd_n_iter))} was passed.')
        # ToDo: Positive integers?

        if (not isinstance(sgd_which_par, str)) | (sgd_which_par not in ['all', 'upper only']):
            raise ValueError('sgd_which_par must be "all" or "upper only". '
                             f'Got {str(sgd_which_par)}.')

        if isinstance(loss_alpha, int):
            loss_alpha = float(loss_alpha)
        if not isinstance(loss_alpha, float):
            raise TypeError('loss_alpha must be of float type. '
                            f'{str(loss_alpha)} of type {str(type(loss_alpha))} was passed.')
        # ToDo: Allowed values for the loss alpha?

        if isinstance(loss_delta_sdp_corr, int):
            loss_delta_sdp_corr = float(loss_delta_sdp_corr)
        if not isinstance(loss_delta_sdp_corr, float):
            raise TypeError('loss_delta_sdp_corr must be of float type. '
                            f'{str(loss_delta_sdp_corr)} of type {str(type(loss_delta_sdp_corr))} was passed.')
        # ToDo: Allowed values for the loss delta sdp corr?

        if isinstance(loss_gamma, int):
            loss_gamma = float(loss_gamma)
        if not isinstance(loss_gamma, float):
            raise TypeError('loss_gamma must be of float type. '
                            f'{str(loss_gamma)} of type {str(type(loss_gamma))} was passed.')
        # ToDo: Allowed values for the loss gamma?

        if isinstance(loss_delta_corr, int):
            loss_delta_corr = float(loss_delta_corr)
        if not isinstance(loss_delta_corr, float):
            raise TypeError('loss_delta_corr must be of float type. '
                            f'{str(loss_delta_corr)} of type {str(type(loss_delta_corr))} was passed.')
        # ToDo: Allowed values for the loss delta corr?

        if (not isinstance(gau_cop_algo, str)) | (gau_cop_algo not in ['sdp', 'ecorr']):
            raise ValueError('gau_cop_algo must be "sdp" or "ecorr". '
                             f'Got {str(gau_cop_algo)}.')

        # fit gaussian copula knockoffs (marginals are fitted and parameters for the decorrelation tree are determined)
        self.fit_gaussian_copula_knockoffs(x_train, marginals=marginals, algo=gau_cop_algo,
                                           vine_structure=vine_structure)

        # select copula families via AIC / MLE
        n_vars = x_train.shape[1]
        n_vars_x2 = n_vars * 2
        u_train = np.full_like(x_train, np.nan)
        for i_var in range(n_vars):
            u_train[:, i_var] = self._marginals[i_var].cdf(x_train[:, i_var])

        # d-vine structure selection happens in fit_gaussian_copula_knockoffs
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
                        self._dvine.copulas[tree - 1][cop - 1] = cop_select(
                            b[:, i - 1], a[:, i + j - 1],
                            families=families, rotations=rotations, indep_test=indep_test)
                    else:
                        self._dvine.copulas[tree - 1][cop - 1] = deepcopy(
                            self._dvine.copulas[tree - 1][cop - 1 - n_vars])

                    if i < n_vars_x2 - j:
                        xx = self._dvine.copulas[tree - 1][cop - 1].hfun(
                            u=b[:, i - 1], v=a[:, i + j - 1])
                    if i > 1:
                        a[:, i + j - 1] = self._dvine.copulas[tree - 1][cop - 1].vfun(
                            u=b[:, i - 1], v=a[:, i + j - 1])
                    if i < n_vars_x2 - j:
                        b[:, i - 1] = xx
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
        loss_obj = KnockoffsLoss(alpha=loss_alpha, delta_sdp_corr=loss_delta_sdp_corr,
                                 gamma=loss_gamma, delta_corr=loss_delta_corr,
                                 mmd_include_diag=True, mmd_sqrt=True)
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

                x_knockoffs[ind_start:ind_end, :], x_knockoffs_deriv = self.generate_par_jacobian(
                    x_test=x_data[ind_start:ind_end, :], knockoff_eps=knockoff_eps[ind_start:ind_end, :],
                    which_par=which_par, return_x_knockoffs=True)

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

    def fit_gaussian_copula_knockoffs(self, x_train, marginals='kde1d',
                                      algo='sdp', vine_structure='1:n'):
        # determine dvine structure / variable order
        n_vars = x_train.shape[1]
        if vine_structure == 'select_tsp_r':
            self.dvine_structure = d_vine_structure_select(x_train, tsp_method='r_tsp')
        elif vine_structure == 'select_tsp_py':
            self.dvine_structure = d_vine_structure_select(x_train, tsp_method='py_tsp')
        else:
            assert vine_structure == '1:n'
            self.dvine_structure = np.arange(n_vars)

        self.fit_marginals(x_train, model=marginals)

        u_train = np.full_like(x_train, np.nan)
        x_gaussian_train = np.full_like(x_train, np.nan)
        for i_var in range(n_vars):
            u_train[:, i_var] = self._marginals[i_var].cdf(x_train[:, i_var])
            x_gaussian_train[:, i_var] = norm.ppf(u_train[:, i_var])

        x_gaussian_train = x_gaussian_train[:, self.dvine_structure]
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
        copulas = [[GaussianCopula().set_par_w_bound_check(rho) for rho in xx] for xx in pcorrs]
        self._dvine = DVineCopula(copulas)
        self.dvine_structure = np.arange(int(self._dvine.n_vars/2))

        return self

    def fit_marginals(self, x_train, model='kde1d'):
        # ToDo May add alternative methods for the marginals (like parameteric distributions)
        n_vars = x_train.shape[1]
        if model == 'kde1d':
            if _has_kde1d:
                self._marginals = [KDE1D().fit(x_train[:, i_var])
                                   for i_var in range(n_vars)]
            else:
                raise ImportError(
                    'To estimate the margins with kde1d the python package rpy2 and the R package kde1d are required.')
        else:
            assert model == 'kde_statsmodels'
            self._marginals = [KDEMultivariateWithInvCdf(x_train[:, i_var], 'c')
                               for i_var in range(n_vars)]

        return self

    def ll(self, x, x_knockoffs):
        n_vars = x.shape[1]

        ll_marginals = 0.
        u_data = np.full_like(x, np.nan)
        u_knockoffs = np.full_like(x_knockoffs, np.nan)
        for i_var in range(n_vars):
            u_data[:, i_var] = self._marginals[i_var].cdf(x[:, i_var])
            u_knockoffs[:, i_var] = self._marginals[i_var].cdf(x_knockoffs[:, i_var])
            ll_marginals += np.log(self._marginals[i_var].pdf(x[:, i_var])).sum()
            ll_marginals += np.log(self._marginals[i_var].pdf(x_knockoffs[:, i_var])).sum()

        u = np.hstack((u_data[:, self.dvine_structure],
                       u_knockoffs[:, self.dvine_structure]))

        ll_copula = self._dvine.ll(u)

        print(ll_copula)
        print(ll_marginals)

        return ll_marginals + ll_copula
