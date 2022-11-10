import numpy as np
import pytest

from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

from vineknockoffs.vine_knockoffs import VineKnockoffs

np.random.seed(3141)
n_obs = 71
n_vars = 4
cov_mat = toeplitz([np.power(0.7, k) for k in range(n_vars)])
x_data = multivariate_normal(mean=np.zeros(n_vars), cov=cov_mat).rvs(n_obs)
vine_ko = VineKnockoffs()


def test_fit_vine_copula_knockoffs_exceptions():
    msg = 'marginals must be "kde1d" or "kde_statsmodels"'
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, marginals='kde')
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, marginals=5)

    msg = 'families must be "all".'
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, families=['Gaussian'])
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, families=5)

    msg = 'rotations must be True or False.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, rotations=90)

    msg = 'indep_test must be True or False.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, indep_test='true')

    msg = 'vine_structure must be "select_tsp_r", "select_tsp_py" or "1:n".'
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, vine_structure='select_tsp')
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, vine_structure=5)

    msg = 'upper_tree_cop_fam_heuristic must be "Gaussian" or "lower tree families".'
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, upper_tree_cop_fam_heuristic='gaussian')
    with pytest.raises(ValueError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, upper_tree_cop_fam_heuristic=True)

    msg = 'sgd_lr must be of float type.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, sgd_lr='1')

    msg = 'sgd_gamma must be of float type.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, sgd_gamma='1')

    msg = 'loss_alpha must be of float type.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, loss_alpha='1')

    msg = 'loss_delta_sdp_corr must be of float type.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, loss_delta_sdp_corr='1')

    msg = 'loss_gamma must be of float type.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, loss_gamma='1')

    msg = 'loss_delta_corr must be of float type.'
    with pytest.raises(TypeError, match=msg):
        vine_ko.fit_vine_copula_knockoffs(x_data, loss_delta_corr='1')

