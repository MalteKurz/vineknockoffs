import numpy as np
from scipy.optimize import root_scalar
from statsmodels.nonparametric.kernel_density import KDEMultivariate


class KDEMultivariateWithInvCdf(KDEMultivariate):
    def ppf(self, x):
        data_min = self.data.min()
        data_max = self.data.max()
        data_range = (data_max - data_min) + 1e-4
        bracket = [data_min - 0.1*data_range, data_max + 0.1*data_range]
        if x.min() < self.cdf(bracket[0:1]):
            bracket[0] -= 0.5*data_range
        if x.max() > self.cdf(bracket[1:2]):
            bracket[1] += 0.5*data_range
        res = np.array([root_scalar(lambda yy: self.cdf(np.array([[yy]]).T) - x[i],
                                    bracket=bracket,
                                    method='brentq',
                                    xtol=1e-12, rtol=1e-12).root for i in range(len(x))])
        return res


class ECDF:

    def __init__(self):
        self._x_sorted = None
        self._n_obs = None

    def fit(self, x):
        self._x_sorted = np.sort(x)
        self._n_obs = len(x)
        return self

    def ppf(self, x):
        ind = int(np.floor((self._n_obs + 1) * x))
        return self._x_sorted[ind]

    def cdf(self, x):
        return np.searchsorted(self._x_sorted, x, side='right') / (self._n_obs+1)

    # def pdf(self, x):
    #     return r_kde1d_pdf_eval(self._kdefit, x)
