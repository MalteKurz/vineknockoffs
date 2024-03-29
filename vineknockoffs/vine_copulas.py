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
        """
        Simulate random observations from a simplified D-vine copula.

        Parameters
        ----------
        n_obs :
            The number of observations to simulate.
            Default is 100.
        w : None or :class:`numpy.ndarray`
            Array of iid standard uniform observations. If None, iid standard uniform observations are drawn.
            Default is None.

        Returns
        -------
        res : :class:`numpy.ndarray`
            Result array.

        Notes
        -----
        Implements Algorithm 2 from Kurz (2022) to simulate from a simplified D-vine copula based on standard uniform
        random variables.

        References
        ----------
        Kurz, M. S. (2022), Vine copula based knockoff generation for high-dimensional controlled variable selection.
        arXiv:`2210.11196 <https://arxiv.org/abs/2210.11196>`__ [stat.ME].
        """
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
            for j in np.arange(1, i):
                tree = i-j
                cop = j
                a[:, j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, j-1], a[:, j-1])
            u[:, i-1] = a[:, i-1]
            if i < self.n_vars:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = i-j
                    cop = j
                    b[:, j-1] = self.copulas[tree-1][cop-1].hfun(b[:, j-1], a[:, j])
        return u

    def sim_d_par(self, which_tree, which_cop, n_obs=100, w=None):
        """
        Simulate random observations from a simplified D-vine copula and compute the derivative with respect to a
        parameter :math:`\\theta_{k,l}`.

        Parameters
        ----------
        which_tree : int
            Determines :math:`\\theta_{k,l}` (copula in which tree?)
        which_cop : int
            Determines :math:`\\theta_{k,l}` (which copula within the tree?)
        n_obs :
            The number of observations to simulate.
            Default is 100.
        w : None or :class:`numpy.ndarray`
            Array of iid standard uniform observations. If None, iid standard uniform observations are drawn.
            Default is None.

        Returns
        -------
        res : :class:`numpy.ndarray`
            Result array(s).

        Notes
        -----
        Implements Algorithm 6 from Kurz (2022) to simulate from a simplified D-vine copula and compute the derivative
        with respect to a parameter :math:`\\theta_{k,l}`.

        .. math::
            U_{1:d} &\\sim C_{1:d},

            \\check{U}_{1:d} &:= \\big(\\partial_{\\theta_{k,l}}U_1, \\ldots, \\partial_{\\theta_{k,l}}U_d)

        References
        ----------
        Kurz, M. S. (2022), Vine copula based knockoff generation for high-dimensional controlled variable selection.
        arXiv:`2210.11196 <https://arxiv.org/abs/2210.11196>`__ [stat.ME].
        """
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
            for j in np.arange(1, i):
                tree = i - j
                cop = j

                if (tree == which_tree) & (cop == which_cop):
                    a_d_par[:, j] = self.copulas[tree-1][cop-1].d_inv_vfun_d_par(b[:, j-1], a[:, j-1])
                    impacted_by_deriv = True
                else:
                    if impacted_by_deriv:
                        d_u = self.copulas[tree-1][cop-1].d_inv_vfun_d_u(b[:, j-1], a[:, j-1])
                        d_v = self.copulas[tree-1][cop-1].d_inv_vfun_d_v(b[:, j-1], a[:, j-1])
                        a_d_par[:, j] = b_d_par[:, j-1] * d_u + a_d_par[:, j-1] * d_v
                    else:
                        a_d_par[:, j] = 0.

                a[:, j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, j-1], a[:, j-1])
            u_d_par[:, i-1] = a_d_par[:, i-1]
            u[:, i-1] = a[:, i-1]
            if i < self.n_vars:
                b_d_par[:, i-1] = a_d_par[:, i-1]
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = i-j
                    cop = j

                    if (tree == which_tree) & (cop == which_cop):
                        d_par = self.copulas[tree-1][cop-1].d_hfun_d_par(b[:, j-1], a[:, j])
                        d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, j-1], a[:, j])
                        b_d_par[:, j-1] = d_par + a_d_par[:, j] * d_v
                    else:
                        if impacted_by_deriv:
                            d_u = self.copulas[tree-1][cop-1].d_hfun_d_u(b[:, j-1], a[:, j])
                            d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, j-1], a[:, j])
                            b_d_par[:, j-1] = b_d_par[:, j-1] * d_u + a_d_par[:, j] * d_v
                        else:
                            b_d_par[:, j-1] = 0.

                    b[:, j-1] = self.copulas[tree-1][cop-1].hfun(b[:, j-1], a[:, j])
        return u_d_par

    def sim_par_jacobian(self, n_obs=100, w=None):
        if w is None:
            w = np.random.uniform(size=(n_obs, self.n_vars))
        res = np.full((w.shape[0], self.n_vars, self.n_pars), np.nan)
        ind_par = 0
        for tree in np.arange(1, self.n_vars):
            for cop in np.arange(1, self.n_vars - tree + 1):
                if not isinstance(self.copulas[tree-1][cop-1], IndepCopula):
                    res[:, :, ind_par] = self.sim_d_par(which_tree=tree, which_cop=cop, w=w)
                    ind_par += 1
        return res

    def get_par_vec(self, from_tree=1):
        par_vec = np.concatenate(
            [cop.par for tree in self.copulas[from_tree - 1:] for cop in tree if cop.par is not None])
        return par_vec

    def set_par_vec(self, par_vec, from_tree=1, assert_to_bounds=False):
        ind_par = 0
        for tree in np.arange(from_tree, self.n_vars):
            for cop in self.copulas[tree - 1]:
                if not isinstance(cop, IndepCopula):
                    assert cop.n_pars == 1
                    if assert_to_bounds:
                        cop._set_par_w_bound_check(par_vec[ind_par])
                    else:
                        cop.par = par_vec[ind_par]
                    ind_par += 1
        return self

    def get_par_vec_w_info(self, from_tree=1):
        # note: from_tree 1-indexed
        n_pars = np.sum([np.sum([cop.n_pars for cop in tree]) for tree in self.copulas[from_tree-1:]])
        par_vec = np.full(n_pars, np.nan)
        lb_vec = np.full(n_pars, np.nan)
        ub_vec = np.full(n_pars, np.nan)
        which_tree = np.full(n_pars, np.nan)
        which_cop = np.full(n_pars, np.nan)
        ind_par = 0
        for tree in np.arange(from_tree, self.n_vars):
            for cop in np.arange(1, self.n_vars - tree + 1):
                cop_n_pars = self.copulas[tree-1][cop-1].n_pars
                if cop_n_pars > 0:
                    assert self.copulas[tree-1][cop-1].n_pars == 1
                    par_vec[ind_par:ind_par+cop_n_pars] = self.copulas[tree-1][cop-1].par
                    lb_vec[ind_par:ind_par+cop_n_pars] = self.copulas[tree-1][cop-1]._par_bounds[0][0]
                    ub_vec[ind_par:ind_par+cop_n_pars] = self.copulas[tree-1][cop-1]._par_bounds[0][1]
                    which_tree[ind_par:ind_par+cop_n_pars] = tree
                    which_cop[ind_par:ind_par+cop_n_pars] = cop
                    ind_par += self.copulas[tree-1][cop-1].n_pars
        par_vec_dict = {'from_tree': from_tree,
                        'n_pars': n_pars,
                        'par_vec': par_vec,
                        'lb_vec': lb_vec,
                        'ub_vec': ub_vec,
                        'which_tree': which_tree,
                        'which_cop': which_cop
                        }
        return par_vec_dict

    def sim_par_jacobian_fast(self, n_obs=100, w=None, w_jacobian=None, from_tree=1, return_u=False):
        if w is None:
            w = np.random.uniform(size=(n_obs, self.n_vars))
        else:
            n_obs = w.shape[0]
        par_vec_dict = self.get_par_vec_w_info(from_tree)
        n_pars = par_vec_dict['n_pars']
        which_tree = par_vec_dict['which_tree']
        which_cop = par_vec_dict['which_cop']

        a = np.full_like(w, np.nan)
        b = np.full_like(w, np.nan)
        u = np.full_like(w, np.nan)

        a_d_par = np.full((n_obs, self.n_vars, n_pars), np.nan)
        b_d_par = np.full_like(a_d_par, np.nan)
        u_d_par = np.full_like(a_d_par, np.nan)
        if w_jacobian is None:
            w_jacobian = np.zeros_like(u_d_par)

        u_d_par[:, 0, :] = w_jacobian[:, 0, :]
        a_d_par[:, 0, :] = w_jacobian[:, 0, :]
        b_d_par[:, 0, :] = w_jacobian[:, 0, :]
        u[:, 0] = w[:, 0]
        a[:, 0] = w[:, 0]
        b[:, 0] = w[:, 0]
        impacted_by_deriv = [False] * n_pars
        for i in np.arange(2, self.n_vars+1):
            a_d_par[:, 0, :] = w_jacobian[:, i-1, :]
            a[:, 0] = w[:, i-1]
            for j in np.arange(1, i):
                tree = i-j
                cop = j

                a[:, j] = self.copulas[tree-1][cop-1].inv_vfun(b[:, j-1], a[:, j-1])
                deriv_computed = False
                d_u = np.nan
                d_v = np.nan
                for i_par in range(n_pars):
                    if (tree == which_tree[i_par]) & (cop == which_cop[i_par]):
                        d_vfun_d_par_eval = self.copulas[tree-1][cop-1].d_vfun_d_par(b[:, j-1], a[:, j])
                        pdf_eval = self.copulas[tree-1][cop-1].pdf(b[:, j-1], a[:, j])
                        d_par = - d_vfun_d_par_eval / pdf_eval
                        d_v = 1. / pdf_eval
                        a_d_par[:, j, i_par] = d_par + a_d_par[:, j-1, i_par] * d_v
                        impacted_by_deriv[i_par] = True
                    else:
                        if impacted_by_deriv[i_par] | \
                                (np.abs(a_d_par[:, j-1, i_par]).sum() > 0.) | \
                                (np.abs(b_d_par[:, j-1, i_par]).sum() > 0.):
                            if not deriv_computed:
                                d_vfun_d_u_eval = self.copulas[tree-1][cop-1].d_vfun_d_u(b[:, j-1], a[:, j])
                                pdf_eval = self.copulas[tree-1][cop-1].pdf(b[:, j-1], a[:, j])
                                d_u = - d_vfun_d_u_eval / pdf_eval
                                d_v = 1. / pdf_eval
                                deriv_computed = True
                            a_d_par[:, j, i_par] = b_d_par[:, j-1, i_par] * d_u + a_d_par[:, j-1, i_par] * d_v
                        else:
                            a_d_par[:, j, i_par] = 0.
            u_d_par[:, i-1, :] = a_d_par[:, i-1, :]
            u[:, i-1] = a[:, i-1]
            if i < self.n_vars:
                b_d_par[:, i-1, :] = a_d_par[:, i-1, :]
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = i-j
                    cop = j

                    deriv_computed = False
                    d_u = np.nan
                    d_v = np.nan
                    for i_par in range(n_pars):
                        if (tree == which_tree[i_par]) & (cop == which_cop[i_par]):
                            d_par = self.copulas[tree-1][cop-1].d_hfun_d_par(b[:, j-1], a[:, j])
                            d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, j-1], a[:, j])
                            b_d_par[:, j-1, i_par] = d_par + a_d_par[:, j, i_par] * d_v
                        else:
                            if impacted_by_deriv[i_par] | \
                                    (np.abs(a_d_par[:, j, i_par]).sum() > 0.) | \
                                    (np.abs(b_d_par[:, j-1, i_par]).sum() > 0.):
                                if not deriv_computed:
                                    d_u = self.copulas[tree-1][cop-1].d_hfun_d_u(b[:, j-1], a[:, j])
                                    d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, j-1], a[:, j])
                                    deriv_computed = True
                                b_d_par[:, j-1, i_par] = b_d_par[:, j-1, i_par] * d_u + a_d_par[:, j, i_par] * d_v
                            else:
                                b_d_par[:, j-1, i_par] = 0.

                    b[:, j-1] = self.copulas[tree-1][cop-1].hfun(b[:, j-1], a[:, j])
        if return_u:
            return u, u_d_par
        else:
            return u_d_par

    def compute_pits(self, u):
        """
        Compute probability integral transforms (PITs) from a simplified D-vine copula.

        Parameters
        ----------
        u : :class:`numpy.ndarray`
            Array of copula observations.

        Returns
        -------
        res : :class:`numpy.ndarray`
            Result array.

        Notes
        -----
        Implements Algorithm 1 from Kurz (2022) to compute the probability integral transforms

        .. math::
            W_{1:d} &:= \\big( U_1, U_{2|1}, U_{3|1:2}, \\ldots, U_{d|1:d-1}\\big)

            &:= \\big( F_1(U_1), F_{2|1}(U_2|U_1), F_{3|1:2}(U_3|U_{1:2}), \\ldots, F_{d|1:d-1}(U_d|U_{1:d-1}) \\big).


        References
        ----------
        Kurz, M. S. (2022), Vine copula based knockoff generation for high-dimensional controlled variable selection.
        arXiv:`2210.11196 <https://arxiv.org/abs/2210.11196>`__ [stat.ME].
        """
        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        w = np.full_like(u, np.nan)

        w[:, 0] = u[:, 0]
        a[:, 0] = u[:, 0]
        b[:, 0] = u[:, 0]

        for i in np.arange(2, self.n_vars+1):
            a[:, i-1] = u[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = i-j
                cop = j
                a[:, j-1] = self.copulas[tree-1][cop-1].vfun(b[:, j-1], a[:, j])
            w[:, i-1] = a[:, 0]
            if i < self.n_vars:
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = i-j
                    cop = j
                    b[:, j-1] = self.copulas[tree-1][cop-1].hfun(b[:, j-1], a[:, j])
        return w

    def compute_pits_d_par(self, which_tree, which_cop, u, return_w=False):
        """
        Compute probability integral transforms (PITs) and their derivative with respect to a parameter
        :math:`\\theta_{k,l}`.

        Parameters
        ----------
        which_tree : int
            Determines :math:`\\theta_{k,l}` (copula in which tree?)
        which_cop : int
            Determines :math:`\\theta_{k,l}` (which copula within the tree?)
        u : :class:`numpy.ndarray`
            Array of copula observations.
        return_w : boolean
            Whether the PITs should also be returned (in addition to the derivatives).

        Returns
        -------
        res : :class:`numpy.ndarray`
            Result array(s).

        Notes
        -----
        Implements Algorithm 5 from Kurz (2022) to compute the probability integral transforms and their derivative with
        respect to a parameter :math:`\\theta_{k,l}`

        .. math::
            W_{1:d} &:= \\big( U_1, U_{2|1}, U_{3|1:2}, \\ldots, U_{d|1:d-1}\\big),

            \\check{W}_{1:d} &:= \\big( \\partial_{\\theta_{k,l}} U_{1}, \\partial_{\\theta_{k,l}} U_{2|1},
            \\partial_{\\theta_{k,l}} U_{3|1:2}, \\ldots, \\partial_{\\theta_{k,l}} U_{d|1:d-1} \\big)

        References
        ----------
        Kurz, M. S. (2022), Vine copula based knockoff generation for high-dimensional controlled variable selection.
        arXiv:`2210.11196 <https://arxiv.org/abs/2210.11196>`__ [stat.ME].
        """
        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        w = np.full_like(u, np.nan)

        a_d_par = np.full_like(u, np.nan)
        b_d_par = np.full_like(u, np.nan)
        w_d_par = np.full_like(u, np.nan)

        w_d_par[:, 0] = 0.
        a_d_par[:, 0] = 0.
        b_d_par[:, 0] = 0.
        w[:, 0] = u[:, 0]
        a[:, 0] = u[:, 0]
        b[:, 0] = u[:, 0]
        impacted_by_deriv = False
        for i in np.arange(2, self.n_vars+1):
            a_d_par[:, i-1] = 0.
            a[:, i-1] = u[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = i-j
                cop = j
                if (tree == which_tree) & (cop == which_cop):
                    a_d_par[:, j-1] = self.copulas[tree-1][cop-1].d_vfun_d_par(b[:, j-1], a[:, j])
                    impacted_by_deriv = True
                else:
                    if impacted_by_deriv:
                        d_u = self.copulas[tree-1][cop-1].d_vfun_d_u(b[:, j-1], a[:, j])
                        d_v = self.copulas[tree-1][cop-1].d_vfun_d_v(b[:, j-1], a[:, j])
                        a_d_par[:, j-1] = b_d_par[:, j-1] * d_u + a_d_par[:, j] * d_v
                    else:
                        a_d_par[:, j-1] = 0.
                a[:, j-1] = self.copulas[tree-1][cop-1].vfun(b[:, j-1], a[:, j])
            w_d_par[:, i-1] = a_d_par[:, 0]
            w[:, i-1] = a[:, 0]
            if i < self.n_vars:
                b_d_par[:, i-1] = a_d_par[:, i-1]
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = i-j
                    cop = j
                    if (tree == which_tree) & (cop == which_cop):
                        b_d_par[:, j-1] = self.copulas[tree-1][cop-1].d_hfun_d_par(b[:, j-1], a[:, j])
                    else:
                        if impacted_by_deriv:
                            d_u = self.copulas[tree-1][cop-1].d_hfun_d_u(b[:, j-1], a[:, j])
                            d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, j-1], a[:, j])
                            b_d_par[:, j-1] = b_d_par[:, j-1] * d_u + a_d_par[:, j] * d_v
                        else:
                            b_d_par[:, j-1] = 0.
                    b[:, j-1] = self.copulas[tree-1][cop-1].hfun(b[:, j-1], a[:, j])
        if return_w:
            return w, w_d_par
        else:
            return w_d_par

    def compute_pits_par_jacobian(self, u):
        res = np.full((u.shape[0], self.n_vars, self.n_pars), np.nan)
        ind_par = 0
        for tree in np.arange(1, self.n_vars):
            for cop in np.arange(1, self.n_vars - tree + 1):
                if not isinstance(self.copulas[tree-1][cop-1], IndepCopula):
                    res[:, :, ind_par] = self.compute_pits_d_par(which_tree=tree, which_cop=cop, u=u)
                    ind_par += 1
        return res

    def compute_pits_par_jacobian_fast(self, u, return_w=False):
        par_vec_dict = self.get_par_vec_w_info(from_tree=1)
        n_pars = par_vec_dict['n_pars']
        which_tree = par_vec_dict['which_tree']
        which_cop = par_vec_dict['which_cop']
        n_obs = u.shape[0]

        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        w = np.full_like(u, np.nan)

        a_d_par = np.full((n_obs, self.n_vars, n_pars), np.nan)
        b_d_par = np.full_like(a_d_par, np.nan)
        w_d_par = np.full_like(a_d_par, np.nan)

        w_d_par[:, 0, :] = 0.
        a_d_par[:, 0, :] = 0.
        b_d_par[:, 0, :] = 0.
        w[:, 0] = u[:, 0]
        a[:, 0] = u[:, 0]
        b[:, 0] = u[:, 0]
        impacted_by_deriv = [False] * n_pars
        for i in np.arange(2, self.n_vars+1):
            a_d_par[:, i-1, :] = 0.
            a[:, i-1] = u[:, i-1]
            for j in np.arange(i-1, 0, -1):
                tree = i-j
                cop = j

                deriv_computed = False
                d_u = np.nan
                d_v = np.nan
                for i_par in range(n_pars):
                    if (tree == which_tree[i_par]) & (cop == which_cop[i_par]):
                        a_d_par[:, j-1, i_par] = self.copulas[tree-1][cop-1].d_vfun_d_par(b[:, j-1], a[:, j])
                        impacted_by_deriv = True
                    else:
                        if impacted_by_deriv:
                            if not deriv_computed:
                                d_u = self.copulas[tree-1][cop-1].d_vfun_d_u(b[:, j-1], a[:, j])
                                d_v = self.copulas[tree-1][cop-1].d_vfun_d_v(b[:, j-1], a[:, j])
                                deriv_computed = True
                            a_d_par[:, j-1, i_par] = b_d_par[:, j-1, i_par] * d_u + a_d_par[:, j, i_par] * d_v
                        else:
                            a_d_par[:, j-1, i_par] = 0.
                a[:, j-1] = self.copulas[tree-1][cop-1].vfun(b[:, j-1], a[:, j])
            w_d_par[:, i-1, :] = a_d_par[:, 0, :]
            w[:, i-1] = a[:, 0]
            if i < self.n_vars:
                b_d_par[:, i-1, :] = a_d_par[:, i-1, :]
                b[:, i-1] = a[:, i-1]
                for j in np.arange(1, i):
                    tree = i-j
                    cop = j

                    deriv_computed = False
                    d_u = np.nan
                    d_v = np.nan
                    for i_par in range(n_pars):
                        if (tree == which_tree[i_par]) & (cop == which_cop[i_par]):
                            b_d_par[:, j-1, i_par] = self.copulas[tree-1][cop-1].d_hfun_d_par(b[:, j-1], a[:, j])
                        else:
                            if impacted_by_deriv:
                                if not deriv_computed:
                                    d_u = self.copulas[tree-1][cop-1].d_hfun_d_u(b[:, j-1], a[:, j])
                                    d_v = self.copulas[tree-1][cop-1].d_hfun_d_v(b[:, j-1], a[:, j])
                                    deriv_computed = True
                                b_d_par[:, j-1, i_par] = b_d_par[:, j-1, i_par] * d_u + a_d_par[:, j, i_par] * d_v
                            else:
                                b_d_par[:, j-1, i_par] = 0.
                    b[:, j-1] = self.copulas[tree-1][cop-1].hfun(b[:, j-1], a[:, j])
        if return_w:
            return w, w_d_par
        else:
            return w_d_par

    def ll(self, u):
        n_vars = u.shape[1]

        a = np.full_like(u, np.nan)
        b = np.full_like(u, np.nan)
        xx = None

        for i in np.arange(1, n_vars):
            a[:, i] = u[:, i]
            b[:, i-1] = u[:, i-1]

        res = 0.
        for j in np.arange(1, n_vars):
            tree = j
            for i in np.arange(1, n_vars-j+1):
                cop = i
                this_cop_ll = self.copulas[tree-1][cop-1].ll(b[:, i-1], a[:, i+j-1]).sum()
                print(f'Tree: {tree}, copula: {cop}, LL: {this_cop_ll}')
                res += this_cop_ll
                if i < n_vars-j:
                    xx = self.copulas[tree-1][cop-1].hfun(b[:, i-1], a[:, i+j-1])
                if i > 1:
                    a[:, i+j-1] = self.copulas[tree-1][cop-1].vfun(b[:, i-1], a[:, i+j-1])
                if i < n_vars-j:
                    b[:, i-1] = xx

        return res

    @classmethod
    def cop_select(cls, u, families='all', rotations=True, indep_test=True):
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
                                                    families=families, rotations=rotations, indep_test=indep_test)
                if i < n_vars-j:
                    xx = copulas[tree-1][cop-1].hfun(u=b[:, i-1], v=a[:, i+j-1])
                if i > 1:
                    a[:, i+j-1] = copulas[tree-1][cop-1].vfun(u=b[:, i-1], v=a[:, i+j-1])
                if i < n_vars-j:
                    b[:, i-1] = xx

        return cls(copulas)
