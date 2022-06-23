import numpy as np
import pytest

from vineknockoffs._utils_copula_families import ClaytonCopula, FrankCopula, GumbelCopula

from _utils_py_vs_r_vinecopula import py_copula_funs_eval, r_copula_funs_eval

rpy2 = pytest.importorskip("rpy2")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

np.random.seed(1111)


@pytest.fixture(scope='module',
                params=['cdf', 'pdf', 'hfun', 'vfun'])
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
                params=[2, 4, 5, 10])
def gumbel_cop_par(request):
    return request.param


def test_clayton_deriv_py_vs_r(clayton_cop_par, fun_type):
    cop_obj = ClaytonCopula()
    n_obs = 231
    data = cop_obj.sim(clayton_cop_par, n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, clayton_cop_par, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               3, clayton_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)


def test_frank_deriv_py_vs_r(frank_cop_par, fun_type):
    cop_obj = FrankCopula()
    n_obs = 231
    data = cop_obj.sim(frank_cop_par, n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, frank_cop_par, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               5, frank_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)


def test_gumbel_deriv_py_vs_r(gumbel_cop_par, fun_type):
    cop_obj = GumbelCopula()
    n_obs = 231
    data = cop_obj.sim(gumbel_cop_par, n_obs)

    res_py = py_copula_funs_eval(data, cop_obj, gumbel_cop_par, fun_type)

    res_r = r_copula_funs_eval(data[:, 0], data[:, 1],
                               4, gumbel_cop_par,
                               fun_type)
    assert np.allclose(res_py,
                       res_r,
                       rtol=1e-9, atol=1e-4)
