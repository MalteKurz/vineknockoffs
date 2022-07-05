import numpy as np
import pytest

from statsmodels.tools.numdiff import approx_fprime

from vineknockoffs.copulas import ClaytonCopula, FrankCopula, GumbelCopula, GaussianCopula, IndepCopula
from vineknockoffs.vine_copulas import DVineCopula

np.random.seed(1111)


@pytest.fixture(scope='module',
                params=[DVineCopula([
                    [ClaytonCopula(4.), ClaytonCopula(3., 90), ClaytonCopula(2.79, 180), ClaytonCopula(5., 270)],
                    [FrankCopula(4.), FrankCopula(-5.), GaussianCopula(-0.23)],
                    [GaussianCopula(0.8), IndepCopula()],
                    [GumbelCopula(6.)]])
                ])
def dvine(request):
    return request.param


def test_sim_numdiff(dvine):
    n_obs = 231
    u_data = np.random.uniform(size=(n_obs, dvine.n_vars))

    res = dvine.sim_par_jacobian(w=u_data)
    par_vec = np.array([cop.par for tree in dvine.copulas for cop in tree if cop.par is not None])

    def sim_for_numdiff(pars, w):
        ind_par = 0
        for tree in range(dvine.n_vars-1):
            for cop in range(dvine.n_vars-1-tree):
                if not isinstance(dvine.copulas[tree][cop], IndepCopula):
                    dvine.copulas[tree][cop]._par = pars[ind_par]
                    ind_par += 1
        return dvine.sim(w=w)

    res_num = np.swapaxes(approx_fprime(par_vec,
                                        sim_for_numdiff,
                                        epsilon=1e-6,
                                        args=(u_data,),
                                        centered=True),
                          0, 1)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-4, atol=1e-3)
