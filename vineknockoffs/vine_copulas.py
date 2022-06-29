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

    def sim(self, n_obs=100, w=None):
        if w is None:
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

        for i in np.arange(2, self.dim+1):
            a[:, 0] = w[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = j
                cop = i-j
                a[:, i-j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, i-j-1], a[:, i-j-1])
            u[:, i-1] = a[:, i-1]
            if i < self.dim:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = j
                    cop = i-j
                    b[:, i-j-1] = self.copulas[tree-1][cop-1].hfun(b[:, i-j-1], a[:, i-j])
        return u

    def compute_pits(self, u):
        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        w = np.full_like(u, np.nan)

        w[:, 0] = u[:, 0]
        a[:, 0] = u[:, 0]
        b[:, 0] = u[:, 0]

        for i in np.arange(2, self.dim+1):
            a[:, i-1] = u[:, i-1]
            for j in np.arange(1, i):
                tree = j
                cop = i-j
                a[:, i-j-1] = self.copulas[tree-1][cop-1].vfun(b[:, i-j-1], a[:, i-j])
            w[:, i-1] = a[:, 0]
            if i < self.dim:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = j
                    cop = i-j
                    b[:, i-j-1] = self.copulas[tree-1][cop-1].hfun(b[:, i-j-1], a[:, i-j])
        return w

    @classmethod
    def cop_select(cls, u, families='all', indep_test=True):
        dim = u.shape[1]
        copulas = [[IndepCopula()] * j for j in np.arange(dim - 1, 0, -1)]

        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        xx = None

        for i in np.arange(1, dim):
            a[:, i] = u[:, i]
            b[:, i-1] = u[:, i-1]

        for j in np.arange(1, dim):
            tree = j
            for i in np.arange(1, dim-j+1):
                cop = i
                copulas[tree-1][cop-1] = cop_select(b[:, i-1], a[:, i+j-1],
                                                    families=families, indep_test=indep_test)
                if i < dim-j:
                    xx = copulas[tree-1][cop-1].hfun(b[:, i-1], a[:, i+j-1])
                if i > 1:
                    a[:, i+j-1] = copulas[tree-1][cop-1].vfun(b[:, i-1], a[:, i+j-1])
                if i < dim-j:
                    b[:, i-1] = xx

        return cls(copulas)
