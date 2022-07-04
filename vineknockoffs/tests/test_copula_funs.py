import numpy as np
import pytest

from statsmodels.tools.numdiff import approx_fprime, approx_hess3

from vineknockoffs.copulas import ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, IndepCopula

np.random.seed(1111)


@pytest.fixture(scope='module',
                params=[ClaytonCopula(4), ClaytonCopula(3, 90), ClaytonCopula(2.79, 180), ClaytonCopula(5, 270),
                        FrankCopula(4), FrankCopula(-5),
                        GaussianCopula(-0.23), GaussianCopula(0.8),
                        GumbelCopula(6), GumbelCopula(3, 90), GumbelCopula(2.79, 180), GumbelCopula(5.2, 270),
                        IndepCopula()])
def copula(request):
    return request.param


def test_hfun(copula):
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.inv_hfun(copula.hfun(data[:, 0], data[:, 1]), data[:, 1])

    assert np.allclose(data[:, 0],
                       res,
                       rtol=1e-9, atol=1e-4)


def test_vfun(copula):
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.inv_vfun(data[:, 0], copula.vfun(data[:, 0], data[:, 1]))

    assert np.allclose(data[:, 1],
                       res,
                       rtol=1e-9, atol=1e-4)


def test_hfun_numdiff(copula):
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.hfun(data[:, 0], data[:, 1])

    def cdf_for_num_diff(v, u):
        return copula.cdf(u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 1],
                                       cdf_for_num_diff,
                                       epsilon=1e-6,
                                       args=(data[i_obs:i_obs+1, 0],),
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_vfun_numdiff(copula):
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.vfun(data[:, 0], data[:, 1])

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 0],
                                       copula.cdf,
                                       epsilon=1e-6,
                                       args=(data[i_obs:i_obs+1, 1],),
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_pdf_numdiff(copula):
    n_obs = 231
    data = copula.sim(n_obs)

    pdf_vals = copula.pdf(data[:, 0], data[:, 1])
    d_vfun_d_u_vals = copula.d_vfun_d_u(data[:, 0], data[:, 1])
    d_hfun_d_v_vals = copula.d_hfun_d_v(data[:, 0], data[:, 1])

    res = np.column_stack((d_vfun_d_u_vals, pdf_vals, pdf_vals, d_hfun_d_v_vals))

    def cdf_for_num_diff(xx):
        return copula.cdf(xx[0:1], xx[1:2])

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs, :] = approx_hess3(data[i_obs, :],
                                         cdf_for_num_diff,
                                         epsilon=1e-6).flatten()

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)
