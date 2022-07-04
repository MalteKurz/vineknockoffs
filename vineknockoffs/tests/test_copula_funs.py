import numpy as np
import pytest

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
