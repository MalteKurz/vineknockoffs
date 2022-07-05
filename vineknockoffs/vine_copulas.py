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

    @property
    def n_pars(self):
        return np.sum([np.sum([cop.n_pars for cop in tree]) for tree in self.copulas])

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

    def sim_d_par(self, which_tree, which_cop, n_obs=100, w=None):
        if w is None:
            w = np.random.uniform(size=(n_obs, self.n_vars))

        a = np.full_like(w, np.nan)
        b = np.full_like(w, np.nan)
        u = np.full_like(w, np.nan)

        a_d_par = np.full_like(w, np.nan)
        b_d_par = np.full_like(w, np.nan)
        u_d_par = np.full_like(w, np.nan)

        u_d_par[:, 0] = 0.
        a_d_par[:, 0] = 0.
        b_d_par[:, 0] = 0.
        u[:, 0] = w[:, 0]
        a[:, 0] = w[:, 0]
        b[:, 0] = w[:, 0]
        impacted_by_deriv = False
        for i in np.arange(2, self.n_vars+1):
            a_d_par[:, 0] = 0.
            a[:, 0] = w[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = j
                cop = i-j

                if (tree == which_tree) & (cop == which_cop):
                    a_d_par[:, i-j] = self.copulas[tree-1][cop-1].d_inv_vfun_d_theta(b[:, i-j-1], a[:, i-j-1])
                    impacted_by_deriv = True
                else:
                    if impacted_by_deriv:
                        d_u = self.copulas[tree-1][cop-1].d_inv_vfun_d_u(b[:, i-j-1], a[:, i-j-1])
                        d_v = self.copulas[tree-1][cop-1].d_inv_vfun_d_v(b[:, i-j-1], a[:, i-j-1])
                        a_d_par[:, i - j] = b[:, i-j-1] * d_u + a[:, i-j-1] * d_v
                    else:
                        a_d_par[:, i - j] = 0.

                a[:, i-j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, i-j-1], a[:, i-j-1])
            u_d_par[:, i-1] = a_d_par[:, i-1]
            u[:, i-1] = a[:, i-1]
            if i < self.n_vars:
                b_d_par[:, i-1] = a_d_par[:, i-1]
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = j
                    cop = i-j

                    if (tree == which_tree) & (cop == which_cop):
                        d_theta = self.copulas[tree-1][cop-1].d_hfun_d_theta(b[:, i-j-1], a[:, i-j])
                        d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, i-j-1], a[:, i-j])
                        b_d_par[:, i - j - 1] = d_theta + a_d_par[:, i-j] * d_v
                    else:
                        if impacted_by_deriv:
                            d_u = self.copulas[tree-1][cop-1].d_hfun_d_u(b[:, i-j-1], a[:, i-j])
                            d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, i-j-1], a[:, i-j])
                            b_d_par[:, i - j - 1] = b_d_par[:, i - j - 1] * d_u + a_d_par[:, i-j] * d_v
                        else:
                            b_d_par[:, i - j - 1] = 0.

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
