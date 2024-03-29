import numpy as np
import pytest

from statsmodels.tools.numdiff import approx_fprime, approx_hess3

from vineknockoffs.copulas import ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, IndepCopula


@pytest.fixture(scope='module',
                params=[ClaytonCopula(4.), ClaytonCopula(3., 90), ClaytonCopula(2.79, 180), ClaytonCopula(5., 270),
                        FrankCopula(4.), FrankCopula(-5.),
                        GaussianCopula(-0.23), GaussianCopula(0.8),
                        GumbelCopula(6.), GumbelCopula(3., 90), GumbelCopula(2.79, 180), GumbelCopula(5.2, 270),
                        IndepCopula()])
def copula(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ClaytonCopula(), ClaytonCopula(rotation=180),
                        FrankCopula(),
                        GaussianCopula(),
                        GumbelCopula(), GumbelCopula(rotation=180)])
def copula_pos_tau(request):
    return request.param


@pytest.fixture(scope='module',
                params=[ClaytonCopula(rotation=90), ClaytonCopula(rotation=270),
                        FrankCopula(),
                        GaussianCopula(),
                        GumbelCopula(rotation=90), GumbelCopula(rotation=270)])
def copula_neg_tau(request):
    return request.param


def test_hfun(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.inv_hfun(copula.hfun(data[:, 0], data[:, 1]), data[:, 1])

    assert np.allclose(data[:, 0],
                       res,
                       rtol=1e-9, atol=1e-4)


def test_vfun(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.inv_vfun(data[:, 0], copula.vfun(data[:, 0], data[:, 1]))

    assert np.allclose(data[:, 1],
                       res,
                       rtol=1e-9, atol=1e-4)


def test_pos_tau2par(copula_pos_tau):
    taus = np.linspace(0.15, 0.85)

    res = np.array([copula_pos_tau.par2tau(copula_pos_tau.tau2par(tau)) for tau in taus])

    assert np.allclose(taus,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_neg_tau2par(copula_neg_tau):
    taus = np.linspace(-0.85, -0.15)

    res = np.array([copula_neg_tau.par2tau(copula_neg_tau.tau2par(tau)) for tau in taus])

    assert np.allclose(taus,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_hfun_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.hfun(data[:, 0], data[:, 1])

    def cdf_for_numdiff(v, u):
        return copula.cdf(u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 1],
                                       cdf_for_numdiff,
                                       epsilon=1e-6,
                                       kwargs={'u': data[i_obs:i_obs+1, 0]},
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_vfun_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.vfun(data[:, 0], data[:, 1])

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(data[i_obs:i_obs+1, 0],
                                       copula.cdf,
                                       epsilon=1e-6,
                                       kwargs={'v': data[i_obs:i_obs+1, 1]},
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_pdf_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    pdf_vals = copula.pdf(data[:, 0], data[:, 1])
    d_vfun_d_u_vals = copula.d_vfun_d_u(data[:, 0], data[:, 1])
    d_hfun_d_v_vals = copula.d_hfun_d_v(data[:, 0], data[:, 1])

    res = np.column_stack((d_vfun_d_u_vals, pdf_vals, pdf_vals, d_hfun_d_v_vals))

    def cdf_for_numdiff(xx):
        return copula.cdf(xx[0:1], xx[1:2])

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs, :] = approx_hess3(data[i_obs, :],
                                         cdf_for_numdiff,
                                         epsilon=1e-6).flatten()

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)


def test_negll_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula._neg_ll_d_par(copula.par, data[:, 0], data[:, 1])

    if isinstance(copula, IndepCopula):
        res_num = 0.
    else:
        res_num = approx_fprime(np.array([copula.par]),
                                copula._neg_ll,
                                epsilon=1e-6,
                                kwargs={'u': data[:, 0], 'v': data[:, 1]},
                                centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)


def test_hfun_d_par_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.d_hfun_d_par(data[:, 0], data[:, 1])

    def hfun_for_numdiff(par, u, v):
        copula._par = par[0]
        return copula.hfun(u, v)

    if isinstance(copula, IndepCopula):
        res_num = np.zeros_like(res)
    else:
        res_num = approx_fprime(np.array([copula.par]),
                                hfun_for_numdiff,
                                epsilon=1e-6,
                                kwargs={'u': data[:, 0], 'v': data[:, 1]},
                                centered=True)

    assert np.allclose(res_num.flatten(),
                       res,
                       rtol=1e-4, atol=1e-3)


def test_vfun_d_par_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.d_vfun_d_par(data[:, 0], data[:, 1])

    def vfun_for_numdiff(par, u, v):
        copula._par = par[0]
        return copula.vfun(u, v)

    if isinstance(copula, IndepCopula):
        res_num = np.zeros_like(res)
    else:
        res_num = approx_fprime(np.array([copula.par]),
                                vfun_for_numdiff,
                                epsilon=1e-6,
                                kwargs={'u': data[:, 0], 'v': data[:, 1]},
                                centered=True)

    assert np.allclose(res_num.flatten(),
                       res,
                       rtol=1e-4, atol=1e-3)


def test_cdf_d_par_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    res = copula.d_cdf_d_par(data[:, 0], data[:, 1])

    def cdf_for_numdiff(par, u, v):
        copula._par = par[0]
        return copula.cdf(u, v)

    if isinstance(copula, IndepCopula):
        res_num = np.zeros_like(res)
    else:
        res_num = approx_fprime(np.array([copula.par]),
                                cdf_for_numdiff,
                                epsilon=1e-6,
                                kwargs={'u': data[:, 0], 'v': data[:, 1]},
                                centered=True)

    assert np.allclose(res_num.flatten(),
                       res,
                       rtol=1e-4, atol=1e-3)


def test_invhfun_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    d_inv_hfun_d_u = copula.d_inv_hfun_d_u(data[:, 0], data[:, 1])
    d_inv_hfun_d_v = copula.d_inv_hfun_d_v(data[:, 0], data[:, 1])
    d_inv_hfun_d_par = copula.d_inv_hfun_d_par(data[:, 0], data[:, 1])

    res = np.column_stack((d_inv_hfun_d_u, d_inv_hfun_d_v, d_inv_hfun_d_par))

    def inv_hfun_for_numdiff_d_v(v, u):
        return copula.inv_hfun(u, v)

    def inv_hfun_for_numdiff_d_par(par, u, v):
        copula._par = par[0]
        return copula.inv_hfun(u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs, 0] = approx_fprime(data[i_obs:i_obs+1, 0],
                                          copula.inv_hfun,
                                          epsilon=1e-6,
                                          kwargs={'v': data[i_obs:i_obs+1, 1]},
                                          centered=True)
        res_num[i_obs, 1] = approx_fprime(data[i_obs:i_obs+1, 1],
                                          inv_hfun_for_numdiff_d_v,
                                          epsilon=1e-6,
                                          kwargs={'u': data[i_obs:i_obs+1, 0]},
                                          centered=True)

    if isinstance(copula, IndepCopula):
        res_num[:, 2] = 0.
    else:
        res_num[:, 2] = approx_fprime(np.array([copula.par]),
                                      inv_hfun_for_numdiff_d_par,
                                      epsilon=1e-6,
                                      kwargs={'u': data[:, 0], 'v': data[:, 1]},
                                      centered=True).flatten()

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)


def test_invvfun_numdiff(copula):
    np.random.seed(3141)
    n_obs = 231
    data = copula.sim(n_obs)

    d_inv_vfun_d_u = copula.d_inv_vfun_d_u(data[:, 0], data[:, 1])
    d_inv_vfun_d_v = copula.d_inv_vfun_d_v(data[:, 0], data[:, 1])
    d_inv_vfun_d_par = copula.d_inv_vfun_d_par(data[:, 0], data[:, 1])

    res = np.column_stack((d_inv_vfun_d_u, d_inv_vfun_d_v, d_inv_vfun_d_par))

    def inv_vfun_for_numdiff_d_v(v, u):
        return copula.inv_vfun(u, v)

    def inv_vfun_for_numdiff_d_par(par, u, v):
        copula._par = par[0]
        return copula.inv_vfun(u, v)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs, 0] = approx_fprime(data[i_obs:i_obs+1, 0],
                                          copula.inv_vfun,
                                          epsilon=1e-6,
                                          kwargs={'v': data[i_obs:i_obs+1, 1]},
                                          centered=True)
        res_num[i_obs, 1] = approx_fprime(data[i_obs:i_obs+1, 1],
                                          inv_vfun_for_numdiff_d_v,
                                          epsilon=1e-6,
                                          kwargs={'u': data[i_obs:i_obs+1, 0]},
                                          centered=True)

    if isinstance(copula, IndepCopula):
        res_num[:, 2] = 0.
    else:
        res_num[:, 2] = approx_fprime(np.array([copula.par]),
                                      inv_vfun_for_numdiff_d_par,
                                      epsilon=1e-6,
                                      kwargs={'u': data[:, 0], 'v': data[:, 1]},
                                      centered=True).flatten()

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)
