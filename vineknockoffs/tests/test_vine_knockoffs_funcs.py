import numpy as np
import pytest

from scipy.stats import norm
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal, bernoulli

from statsmodels.tools.numdiff import approx_fprime

from vineknockoffs.copulas import ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, IndepCopula
from vineknockoffs.vine_copulas import DVineCopula
from vineknockoffs.vine_knockoffs import VineKnockoffs
from vineknockoffs.knockoffs import KockoffsLoss

from vineknockoffs._utils_gaussian_knockoffs import sdp_solver

np.random.seed(1111)


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


def test_generate_numdiff(which_par):
    n_obs = 71
    n_vars = 4
    # u_data = dvine.sim(n_obs)
    # x_data = norm.ppf(u_data)

    cov_mat = toeplitz([np.power(0.7, k) for k in range(n_vars)])
    x_data = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)

    vine_ko = VineKnockoffs()
    vine_ko.fit_vine_copula_knockoffs(x_data, indep_test=False)
    # vine_ko.fit_gaussian_knockoffs(x_data)
    for t in vine_ko._dvine.copulas:
        for c in t:
            if isinstance(c, GaussianCopula):
                if c.par > 0.99:
                    c._par = 0.99
                elif c.par < -0.99:
                    c._par = -0.99

    # u_test = dvine.sim(n_obs)
    # x_test = norm.ppf(u_test)
    x_test = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)
    knockoff_eps = np.random.uniform(size=(n_obs, n_vars))

    res = vine_ko.generate_par_jacobian(x_test=x_test, knockoff_eps=knockoff_eps, which_par=which_par)

    if which_par == 'upper only':
        start_tree = n_vars
    else:
        start_tree = 1
    par_vec = np.array([cop.par for tree in vine_ko._dvine.copulas[start_tree-1:] for cop in tree if cop.par is not None])

    def generate_for_numdiff(pars, from_tree, xx_test, ko_eps):
        ind_par = 0
        for tree in np.arange(from_tree, vine_ko._dvine.n_vars):
            for cop in np.arange(1, vine_ko._dvine.n_vars-tree+1):
                if not isinstance(vine_ko._dvine.copulas[tree-1][cop-1], IndepCopula):
                    vine_ko._dvine.copulas[tree-1][cop-1]._par = pars[ind_par]
                    ind_par += 1
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


def test_loss_numdiff(which_par):
    n_obs = 71
    n_vars = 4

    cov_mat = toeplitz([np.power(0.7, k) for k in range(n_vars)])
    x_data = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)

    vine_ko = VineKnockoffs()
    vine_ko.fit_vine_copula_knockoffs(x_data)
    # vine_ko.fit_gaussian_knockoffs(x_data)
    for t in vine_ko._dvine.copulas:
        for c in t:
            if isinstance(c, GaussianCopula):
                if c.par > 0.99:
                    c._par = 0.99
                elif c.par < -0.99:
                    c._par = -0.99

    x_test = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)
    knockoff_eps = np.random.uniform(size=(n_obs, n_vars))

    x_knockoffs = vine_ko.generate(x_test=x_test, knockoff_eps=knockoff_eps)
    x_knockoffs_deriv = vine_ko.generate_par_jacobian(x_test=x_test, knockoff_eps=knockoff_eps, which_par=which_par)

    loss = KockoffsLoss(delta_corr=1.)
    swap_inds = np.arange(0, n_vars)[bernoulli.rvs(0.5, size=n_vars) == 1]
    corr_mat = np.corrcoef(x_test.transpose())
    sdp_corr = 1. - sdp_solver(corr_mat)
    res = loss.deriv(x=x_test, x_knockoffs=x_knockoffs, x_knockoffs_deriv=x_knockoffs_deriv,
                     swap_inds=swap_inds, sdp_corr=sdp_corr)[0]

    if which_par == 'upper only':
        start_tree = n_vars
    else:
        start_tree = 1
    par_vec = np.array([cop.par for tree in vine_ko._dvine.copulas[start_tree-1:] for cop in tree if cop.par is not None])

    def generate_for_numdiff(pars, from_tree, xx_test, ko_eps):
        ind_par = 0
        for tree in np.arange(from_tree, vine_ko._dvine.n_vars):
            for cop in np.arange(1, vine_ko._dvine.n_vars-tree+1):
                if not isinstance(vine_ko._dvine.copulas[tree-1][cop-1], IndepCopula):
                    vine_ko._dvine.copulas[tree-1][cop-1]._par = pars[ind_par]
                    ind_par += 1
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
