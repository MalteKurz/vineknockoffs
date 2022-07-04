import numpy as np
import pytest

from vineknockoffs.copulas import ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, IndepCopula

from _utils_py_vs_r_vinecopula import py_copula_funs_eval, r_copula_funs_eval

rpy2 = pytest.importorskip("rpy2")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

np.random.seed(1111)


@pytest.fixture(scope='module',
                params=['cdf', 'pdf', 'hfun', 'vfun',
                        'd_hfun_d_theta', 'd_vfun_d_theta', 'd_hfun_d_v', 'd_vfun_d_u'])
def fun_type(request):
    return request.param


@pytest.fixture(scope='module',
                params=[2, 4, 5, 10])
def clayton_cop_par(request):
    return request.param


@pytest.fixture(scope='module',
                params=[2, 4, 5, 10,
                        -2, -4, -5, -10])
def frank_cop_par(request):
    return request.param


@pytest.fixture(scope='module',
                params=[-0.2, 0.2, 0.5, 0.75])
def gaussian_cop_par(request):
    return request.param


@pytest.fixture(scope='module',
                params=[2, 4, 5, 10])
def gumbel_cop_par(request):
    return request.param


def test_clayton_deriv_py_vs_r(clayton_cop_par, fun_type):
    cop_obj = ClaytonCopula(clayton_cop_par)
    n_obs = 231
    data = cop_obj.sim(n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               3, clayton_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)


def test_frank_deriv_py_vs_r(frank_cop_par, fun_type):
    cop_obj = FrankCopula(frank_cop_par)
    n_obs = 231
    data = cop_obj.sim(n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               5, frank_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)


def test_gaussian_deriv_py_vs_r(gaussian_cop_par, fun_type):
    cop_obj = GaussianCopula(gaussian_cop_par)
    n_obs = 231
    data = cop_obj.sim(n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               1, gaussian_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)


def test_gumbel_deriv_py_vs_r(gumbel_cop_par, fun_type):
    cop_obj = GumbelCopula(gumbel_cop_par)
    n_obs = 231
    data = cop_obj.sim(n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               4, gumbel_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)


def test_indep_deriv_py_vs_r(fun_type):
    cop_obj = IndepCopula()
    n_obs = 231
    data = cop_obj.sim(n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, fun_type)

    cop_par = 0  # ignored
    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               0, cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)
