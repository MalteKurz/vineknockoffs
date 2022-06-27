import numpy as np
from .copulas import cop_select, IndepCopula


class DVineCopula:

    def __init__(self, copulas):
        self._copulas = copulas

    @property
    def copulas(self):
        return self._copulas

    @property
    def dim(self):
        return len(self.copulas) + 1

    def sim(self, n_obs=100):
        w = np.random.uniform(size=(n_obs, self.dim))

        a = np.full_like(w, np.nan)
        b = np.full_like(w, np.nan)
        u = np.full_like(w, np.nan)

        u[:, 0] = w[:, 0]
        a[:, 0] = w[:, 0]
        b[:, 0] = w[:, 0]

        # for i in np.arange(1, self.dim):
        #     a[:, 0] = w[:, i]
        #     for j in np.arange(0, i-1):
        #         tree = i-j
        #         cop = j
        #         a[:, j+1] = self.copulas[tree][cop].vfun(b[:, j], a[:, j])
        #     u[:, i] = a[:, i]
        #     b[:, i] = a[:, i]
        #     for j in np.arange(i-2, -1, -1):
        #         tree = i-j
        #         cop = j
        #         b[:, j] = self.copulas[tree][cop].hfun(b[:, j], a[:, j+1])

        for i in np.arange(1, self.dim):
            a[:, 0] = w[:, i]
            for j in np.arange(i-1, -1, -1):
                tree = j
                cop = i-j-1
                a[:, i-j] = self.copulas[tree][cop].vfun(b[:, i-j-1], a[:, i-j-1])
            u[:, i] = a[:, i]
            b[:, i] = a[:, i]
            for j in np.arange(0, i):
                tree = j
                cop = i-j-1
                b[:, i-j-1] = self.copulas[tree][cop].hfun(b[:, i-j-1], a[:, i-j])
        return u

    @classmethod
    def cop_select(cls, u, families='all', indep_test=True):
        dim = np.shape(u)[1]
        copulas = [[IndepCopula()] * j for j in np.arange(dim - 1, 0, -1)]

        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        xx = None

        for i in np.arange(0, dim-1):
            a[:, i+1] = u[:, i+1]
            b[:, i] = u[:, i]

        for j in np.arange(0, dim-1):
            tree = j
            for i in np.arange(0, dim-j-1):
                cop = i
                copulas[tree][cop] = cop_select(b[:, i], a[:, i+j+1],
                                                families=families, indep_test=indep_test)
                if i < dim-j-1:
                    xx = copulas[tree][cop].hfun(b[:, i], a[:, i+j+1])
                if i > 0:
                    a[:, i + j+1] = copulas[tree][cop].vfun(b[:, i], a[:, i+j+1])
                if i < dim-j-1:
                    b[:, i] = xx

        return cls(copulas)
