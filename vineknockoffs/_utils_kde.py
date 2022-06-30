import numpy as np
from scipy.optimize import root_scalar
from statsmodels.nonparametric.kernel_density import KDEMultivariate


class KDEMultivariateWithInvCdf(KDEMultivariate):
    def ppf(self, x):
        data_min = self.data.min()
        data_max = self.data.max()
        data_range = (data_max - data_min) + 1e-4
        bracket = [data_min - 0.1*data_range, data_max + 0.1*data_range]
        res = np.array([root_scalar(lambda yy: self.cdf(np.array([[yy]]).T) - x[i],
                                    bracket=bracket,
                                    method='brentq',
                                    xtol=1e-12, rtol=1e-12).root for i in range(len(x))])
        return res
