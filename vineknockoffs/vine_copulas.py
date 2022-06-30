import numpy as np

from .copulas import cop_select, IndepCopula


class DVineCopula:

    def __init__(self, copulas):
        self._copulas = copulas

    @property
    def copulas(self):
        return self._copulas

    @property
    def n_vars(self):
        return len(self.copulas) + 1

    def sim(self, n_obs=100, w=None):
        if w is None:
            w = np.random.uniform(size=(n_obs, self.n_vars))

        a = np.full_like(w, np.nan)
        b = np.full_like(w, np.nan)
        u = np.full_like(w, np.nan)

        u[:, 0] = w[:, 0]
        a[:, 0] = w[:, 0]
        b[:, 0] = w[:, 0]

        for i in np.arange(2, self.n_vars+1):
            a[:, 0] = w[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = j
                cop = i-j
                a[:, i-j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, i-j-1], a[:, i-j-1])
            u[:, i-1] = a[:, i-1]
            if i < self.n_vars:
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

        for i in np.arange(2, self.n_vars+1):
            a[:, i-1] = u[:, i-1]
            for j in np.arange(1, i):
                tree = j
                cop = i-j
                a[:, i-j-1] = self.copulas[tree-1][cop-1].vfun(b[:, i-j-1], a[:, i-j])
            w[:, i-1] = a[:, 0]
            if i < self.n_vars:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = j
                    cop = i-j
                    b[:, i-j-1] = self.copulas[tree-1][cop-1].hfun(b[:, i-j-1], a[:, i-j])
        return w

    @classmethod
    def cop_select(cls, u, families='all', indep_test=True):
        n_vars = u.shape[1]
        copulas = [[IndepCopula()] * j for j in np.arange(n_vars - 1, 0, -1)]

        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        xx = None

        for i in np.arange(1, n_vars):
            a[:, i] = u[:, i]
            b[:, i-1] = u[:, i-1]

        for j in np.arange(1, n_vars):
            tree = j
            for i in np.arange(1, n_vars-j+1):
                cop = i
                copulas[tree-1][cop-1] = cop_select(b[:, i-1], a[:, i+j-1],
                                                    families=families, indep_test=indep_test)
                if i < n_vars-j:
                    xx = copulas[tree-1][cop-1].hfun(b[:, i-1], a[:, i+j-1])
                if i > 1:
                    a[:, i+j-1] = copulas[tree-1][cop-1].vfun(b[:, i-1], a[:, i+j-1])
                if i < n_vars-j:
                    b[:, i-1] = xx

        return cls(copulas)
