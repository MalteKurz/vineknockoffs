import numpy as np
import pytest

from scipy.stats import norm
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal, bernoulli

from statsmodels.tools.numdiff import approx_fprime

from vineknockoffs.copulas import ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, IndepCopula
from vineknockoffs.vine_copulas import DVineCopula
from vineknockoffs.vine_knockoffs import VineKnockoffs
from vineknockoffs.knockoffs import KnockoffsLoss

from vineknockoffs._utils_gaussian_knockoffs import sdp_solver

try:
    from rpy2 import robjects
except ImportError:
    _has_rpy2 = False
else:
    _has_rpy2 = True


if _has_rpy2:
    vine_structures = ['select_tsp_r', 'select_tsp_py']
    marginal_models = ['kde1d', 'kde_statsmodels']
else:
    vine_structures = ['select_tsp_py']
    marginal_models = ['kde_statsmodels']


# @pytest.fixture(scope='module',
#                 params=[DVineCopula([
#                     [ClaytonCopula(4.), ClaytonCopula(2.79, 180)],
#                     [GaussianCopula(-0.23)]]),
#                     DVineCopula([
#                         [ClaytonCopula(4.), ClaytonCopula(2.79, 180), ClaytonCopula(5., 270)],
#                         [FrankCopula(4.), GaussianCopula(-0.23)],
#                         [GumbelCopula(6.)]]),
#                 ])
# def dvine(request):
#     return request.param

@pytest.fixture(scope='module',
                params=['upper only', 'all'])
def which_par(request):
    return request.param


@pytest.fixture(scope='module',
                params=vine_structures)
def vine_structure(request):
    return request.param


@pytest.fixture(scope='module',
                params=marginal_models)
def marginals(request):
    return request.param


def test_generate_numdiff(which_par, vine_structure, marginals):
    np.random.seed(3141)
    n_obs = 71
    n_vars = 4
    # u_data = dvine.sim(n_obs)
    # x_data = norm.ppf(u_data)

    cov_mat = toeplitz([np.power(0.7, k) for k in range(n_vars)])
    x_data = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)

    vine_ko = VineKnockoffs()
    vine_ko.fit_vine_copula_knockoffs(x_data, sgd=False, vine_structure=vine_structure, marginals=marginals)
    # vine_ko.fit_gaussian_knockoffs(x_data)

    # u_test = dvine.sim(n_obs)
    # x_test = norm.ppf(u_test)
    x_test = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)
    knockoff_eps = np.random.uniform(size=(n_obs, n_vars))

    res = vine_ko.generate_par_jacobian(x_test=x_test, knockoff_eps=knockoff_eps, which_par=which_par)

    if which_par == 'upper only':
        start_tree = n_vars
    else:
        start_tree = 1
    par_vec = vine_ko._dvine.get_par_vec(from_tree=start_tree)

    def generate_for_numdiff(pars, from_tree, xx_test, ko_eps):
        vine_ko._dvine.set_par_vec(par_vec=pars, from_tree=from_tree)
        return vine_ko.generate(x_test=xx_test, knockoff_eps=ko_eps)

    res_num = np.swapaxes(approx_fprime(par_vec,
                                        generate_for_numdiff,
                                        epsilon=1e-6,
                                        kwargs={'from_tree': start_tree,
                                                'xx_test': x_test,
                                                'ko_eps': knockoff_eps},
                                        centered=True),
                          0, 1)

    assert np.allclose(res_num.astype('float64'),
                       res,
                       rtol=1e-4, atol=1e-3)


def test_loss_numdiff(which_par, vine_structure, marginals):
    np.random.seed(3141)
    n_obs = 71
    n_vars = 4

    cov_mat = toeplitz([np.power(0.7, k) for k in range(n_vars)])
    x_data = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)

    vine_ko = VineKnockoffs()
    vine_ko.fit_vine_copula_knockoffs(x_data, sgd=False, vine_structure=vine_structure, marginals=marginals)
    # vine_ko.fit_gaussian_knockoffs(x_data)

    x_test = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)
    knockoff_eps = np.random.uniform(size=(n_obs, n_vars))

    x_knockoffs = vine_ko.generate(x_test=x_test, knockoff_eps=knockoff_eps)
    x_knockoffs_deriv = vine_ko.generate_par_jacobian(x_test=x_test, knockoff_eps=knockoff_eps, which_par=which_par)

    loss = KnockoffsLoss(delta_corr=1., mmd_include_diag=True, mmd_sqrt=True)
    swap_inds = np.arange(0, n_vars)[bernoulli.rvs(0.5, size=n_vars) == 1]
    corr_mat = np.corrcoef(x_test.transpose())
    sdp_corr = 1. - sdp_solver(corr_mat)
    res = loss.deriv(x=x_test, x_knockoffs=x_knockoffs, x_knockoffs_deriv=x_knockoffs_deriv,
                     swap_inds=swap_inds, sdp_corr=sdp_corr)[0]

    if which_par == 'upper only':
        start_tree = n_vars
    else:
        start_tree = 1
    par_vec = vine_ko._dvine.get_par_vec(from_tree=start_tree)

    def generate_for_numdiff(pars, from_tree, xx_test, ko_eps):
        vine_ko._dvine.set_par_vec(par_vec=pars, from_tree=from_tree)
        xx_knockoffs = vine_ko.generate(x_test=xx_test, knockoff_eps=ko_eps)
        xx_res = loss.eval(x=x_test, x_knockoffs=xx_knockoffs,
                           swap_inds=swap_inds, sdp_corr=sdp_corr)
        return xx_res[0]

    res_num = approx_fprime(par_vec,
                            generate_for_numdiff,
                            epsilon=1e-6,
                            kwargs={'from_tree': start_tree,
                                    'xx_test': x_test,
                                    'ko_eps': knockoff_eps},
                            centered=True)

    assert np.allclose(res_num.astype('float64'),
                       res,
                       rtol=1e-4, atol=1e-3)
