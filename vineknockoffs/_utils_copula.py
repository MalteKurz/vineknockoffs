import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import kendalltau
from scipy.optimize import fmin_l_bfgs_b


class Copula(ABC):
    _theta_bounds = None

    def __init__(self):
        return

    def mle_est(self, u, v):
        tau, _ = kendalltau(u, v)
        theta_0 = self.tau2par(tau)
        theta_hat, _, _ = fmin_l_bfgs_b(self.neg_ll,
                                        theta_0,
                                        self.neg_ll_deriv_theta,
                                        (u, v),
                                        bounds=self._theta_bounds)
        return theta_hat

    @staticmethod
    @abstractmethod
    def tau2par(tau):
        pass

    @staticmethod
    @abstractmethod
    def neg_ll(theta, u, v):
        pass

    @staticmethod
    @abstractmethod
    def neg_ll_deriv_theta(theta, u, v):
        pass

    @staticmethod
    @abstractmethod
    def inv_h_fun(theta, u, v):
        pass

    def sim(self, theta, n_obs=100):
        u = np.random.uniform(size=(n_obs, 2))
        u[:, 0] = self.inv_h_fun(theta, u[:, 0], u[:, 1])
        return
