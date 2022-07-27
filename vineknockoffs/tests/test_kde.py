import numpy as np

from statsmodels.tools.numdiff import approx_fprime

from vineknockoffs._utils_kde import KDEMultivariateWithInvCdf


def test_inv_cdf():
    np.random.seed(3141)
    n_obs = 124
    data = np.random.normal(size=n_obs)
    marginal = KDEMultivariateWithInvCdf(data, 'c')

    res = marginal.ppf(marginal.cdf(data))

    assert np.allclose(data,
                       res,
                       rtol=1e-9, atol=1e-4)


def test_inv_cdf_numdiff():
    np.random.seed(3141)
    n_obs = 124
    data = np.random.normal(size=n_obs)
    marginal = KDEMultivariateWithInvCdf(data, 'c')
    u_data = marginal.cdf(data)

    res = 1. / marginal.pdf(data)

    def inv_cdf_for_numdiff(u):
        return marginal.ppf(u)

    res_num = np.full_like(res, np.nan)
    for i_obs in range(n_obs):
        res_num[i_obs] = approx_fprime(u_data[i_obs:i_obs+1],
                                       inv_cdf_for_numdiff,
                                       epsilon=1e-6,
                                       centered=True)

    assert np.allclose(res_num,
                       res,
                       rtol=1e-9, atol=1e-4)
