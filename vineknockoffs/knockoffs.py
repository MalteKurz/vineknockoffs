import numpy as np
from scipy.stats import bernoulli
from scipy.spatial.distance import cdist

from ._utils_gaussian_knockoffs import sdp_solver


class KnockoffsLoss:

    def __init__(self, alpha=1., delta_sdp_corr=1., gamma=1., delta_corr=0.,
                 mmd_include_diag=False, mmd_sqrt=False):
        self._alpha = alpha
        self._delta_sdp_corr = delta_sdp_corr
        self._gamma = gamma
        self._delta_corr = delta_corr
        self._mmd_include_diag = mmd_include_diag
        self._mmd_sqrt = mmd_sqrt

    @property
    def alpha(self):
        return self._alpha

    @property
    def delta_sdp_corr(self):
        return self._delta_sdp_corr

    @property
    def gamma(self):
        return self._gamma

    @property
    def delta_corr(self):
        return self._delta_corr

    @property
    def mmd_include_diag(self):
        return self._mmd_include_diag

    @property
    def mmd_sqrt(self):
        return self._mmd_sqrt

    def eval(self, x, x_knockoffs, swap_inds=None, sdp_corr=None):
        if sdp_corr is None:
            corr_mat = np.corrcoef(x.transpose())
            sdp_corr = 1. - sdp_solver(corr_mat)

        n_obs = x.shape[0]
        dim_x = x.shape[1]

        if swap_inds is None:
            swap_inds = np.arange(0, dim_x)[bernoulli.rvs(0.5, size=dim_x) == 1]

        # Loss part 1: First and second moments
        mu_data = x.mean(axis=0)
        mu_knockoffs = x_knockoffs.mean(axis=0)

        x_dm = x - mu_data
        x_knockoffs_dm = x_knockoffs - mu_knockoffs

        sigma_x = x_dm.T @ x_dm / n_obs
        sigma_x_knockoffs = x_knockoffs_dm.T @ x_knockoffs_dm / n_obs
        sigma_x_x_knockoffs = x_knockoffs_dm.T @ x_dm / n_obs

        denom = np.power(sigma_x, 2).sum()

        loss_1m = np.power(mu_data - mu_knockoffs,  2).sum()

        loss_2m_diag = np.power(sigma_x - sigma_x_knockoffs, 2).sum()

        m_mat = np.ones_like(sigma_x)
        np.fill_diagonal(m_mat, 0.)
        loss_2m_offdiag = np.power(m_mat * (sigma_x - sigma_x_x_knockoffs), 2).sum()

        loss_moments = loss_1m / dim_x + loss_2m_diag / denom + loss_2m_offdiag / denom

        # Loss part 2: Correlations
        var_x = sigma_x.diagonal()
        var_x_knockoffs = sigma_x_knockoffs.diagonal()
        cov_x_x_knockoffs = sigma_x_x_knockoffs.diagonal()
        corr_x_x_knockoffs = cov_x_x_knockoffs / np.sqrt(var_x * var_x_knockoffs)
        loss_sdp_corr = np.power(sdp_corr - corr_x_x_knockoffs, 2).sum() / dim_x
        loss_corr = np.power(corr_x_x_knockoffs, 2).sum() / dim_x

        alphas = np.array([1., 2., 4., 8., 16., 32., 64., 128.])
        n_obs_half = int(np.floor(n_obs / 2.))
        x_part1 = x[:n_obs_half, :]
        x_part2 = x[n_obs_half:, :]
        x_knockoffs_part1 = x_knockoffs[:n_obs_half, :]
        x_knockoffs_part2 = x_knockoffs[n_obs_half:, :]

        z1 = np.hstack((x_part1, x_knockoffs_part1))
        # full swap
        z2 = np.hstack((x_knockoffs_part2, x_part2))
        loss_mmd_full = self._loss_mmd(z1, z2, alphas, self.mmd_include_diag, self.mmd_sqrt)

        # partial swap
        z3 = np.hstack((x_part2, x_knockoffs_part2))
        z3[:, swap_inds] = x_knockoffs_part2[:, swap_inds]
        z3[:, swap_inds + dim_x] = x_part2[:, swap_inds]
        loss_mmd_partial = self._loss_mmd(z1, z3, alphas, self.mmd_include_diag, self.mmd_sqrt)

        loss_mmd_total = loss_mmd_full + loss_mmd_partial

        loss = self.alpha * loss_moments + self.delta_sdp_corr * loss_sdp_corr \
            + self.gamma * loss_mmd_total + self.delta_corr * loss_corr
        return loss, loss_moments, loss_sdp_corr, loss_mmd_total, loss_corr

    def deriv(self, x, x_knockoffs, x_knockoffs_deriv, swap_inds=None, sdp_corr=None):
        if sdp_corr is None:
            corr_mat = np.corrcoef(x.transpose())
            sdp_corr = 1. - sdp_solver(corr_mat)

        n_obs = x.shape[0]
        dim_x = x.shape[1]
        n_pars = x_knockoffs_deriv.shape[2]

        if swap_inds is None:
            swap_inds = np.arange(0, dim_x)[bernoulli.rvs(0.5, size=dim_x) == 1]

        # Loss part 1: First and second moments
        mu_data = x.mean(axis=0)
        mu_knockoffs = x_knockoffs.mean(axis=0)

        x_dm = x - mu_data
        x_knockoffs_dm = x_knockoffs - mu_knockoffs

        sigma_x = x_dm.T @ x_dm / n_obs
        sigma_x_knockoffs = x_knockoffs_dm.T @ x_knockoffs_dm / n_obs
        sigma_x_x_knockoffs = x_knockoffs_dm.T @ x_dm / n_obs

        denom = np.power(sigma_x, 2).sum()

        m_mat = np.ones_like(sigma_x)
        np.fill_diagonal(m_mat, 0.)

        # Loss part 2: Correlations
        var_x = sigma_x.diagonal()
        var_x_knockoffs = sigma_x_knockoffs.diagonal()
        cov_x_x_knockoffs = sigma_x_x_knockoffs.diagonal()
        corr_x_x_knockoffs = cov_x_x_knockoffs / np.sqrt(var_x * var_x_knockoffs)

        # Derivatives for loss part 1 & 2
        loss_moments_deriv = np.zeros(n_pars)
        loss_sdp_corr_deriv = np.zeros(n_pars)
        loss_corr_deriv = np.zeros(n_pars)
        for i_par in range(n_pars):
            mu_derivs = x_knockoffs_deriv[:, :, i_par].mean(axis=0)
            x_knockoffs_deriv_dm = x_knockoffs_deriv[:, :, i_par] - mu_derivs

            sigma_x_knockoffs_deriv = x_knockoffs_deriv_dm.T @ x_knockoffs_dm / n_obs \
                + x_knockoffs_dm.T @ x_knockoffs_deriv_dm / n_obs
            sigma_x_x_knockoffs_deriv = x_knockoffs_deriv_dm.T @ x_dm / n_obs

            loss_1m_deriv = -2 * ((mu_data - mu_knockoffs) * mu_derivs).sum()

            loss_2m_diag_deriv = -2 * ((sigma_x - sigma_x_knockoffs) * sigma_x_knockoffs_deriv).sum()
            loss_2m_offdiag_deriv = -2 * (m_mat * ((sigma_x - sigma_x_x_knockoffs) * sigma_x_x_knockoffs_deriv)).sum()

            loss_moments_deriv[i_par] = loss_1m_deriv / dim_x + loss_2m_diag_deriv / denom \
                + loss_2m_offdiag_deriv / denom

            var_x_knockoffs_deriv = sigma_x_knockoffs_deriv.diagonal()
            cov_x_x_knockoffs_deriv = sigma_x_x_knockoffs_deriv.diagonal()
            corr_x_x_knockoffs_deriv = (cov_x_x_knockoffs_deriv * var_x * var_x_knockoffs
                                        - 0.5 * cov_x_x_knockoffs * var_x_knockoffs_deriv * var_x) \
                / np.power(var_x * var_x_knockoffs, 1.5)

            loss_sdp_corr_deriv[i_par] = -2 * ((sdp_corr - corr_x_x_knockoffs) * corr_x_x_knockoffs_deriv).sum() / dim_x
            loss_corr_deriv[i_par] = 2 * (corr_x_x_knockoffs * corr_x_x_knockoffs_deriv).sum() / dim_x

        alphas = np.array([1., 2., 4., 8., 16., 32., 64., 128.])
        n_obs_half = int(np.floor(n_obs / 2.))
        x_part1 = x[:n_obs_half, :]
        x_part2 = x[n_obs_half:, :]
        x_knockoffs_part1 = x_knockoffs[:n_obs_half, :]
        x_knockoffs_part2 = x_knockoffs[n_obs_half:, :]

        z1 = np.hstack((x_part1, x_knockoffs_part1))
        # full swap
        z2 = np.hstack((x_knockoffs_part2, x_part2))

        # partial swap
        z3 = np.hstack((x_part2, x_knockoffs_part2))
        z3[:, swap_inds] = x_knockoffs_part2[:, swap_inds]
        z3[:, swap_inds + dim_x] = x_part2[:, swap_inds]

        x_deriv_part1 = np.zeros_like(x_part1)
        x_deriv_part2 = np.zeros_like(x_part2)

        z1_deriv = np.full((n_obs_half, 2*dim_x, n_pars), np.nan)
        z2_deriv = np.full((n_obs - n_obs_half, 2*dim_x, n_pars), np.nan)
        z3_deriv = np.full_like(z2_deriv, np.nan)

        for i_par in range(n_pars):
            x_knockoffs_deriv_part1 = x_knockoffs_deriv[:n_obs_half, :, i_par]
            x_knockoffs_deriv_part2 = x_knockoffs_deriv[n_obs_half:, :, i_par]

            z1_deriv[:, :, i_par] = np.hstack((x_deriv_part1, x_knockoffs_deriv_part1))
            z2_deriv[:, :, i_par] = np.hstack((x_knockoffs_deriv_part2, x_deriv_part2))
            z3_deriv[:, :, i_par] = np.hstack((x_deriv_part2, x_knockoffs_deriv_part2))
            z3_deriv[:, swap_inds, i_par] = x_knockoffs_deriv_part2[:, swap_inds]
            z3_deriv[:, swap_inds+dim_x, i_par] = x_deriv_part2[:, swap_inds]

        loss_mmd_full_deriv = self._loss_mmd_deriv(z1, z2, z1_deriv, z2_deriv, alphas,
                                                   self.mmd_include_diag, self.mmd_sqrt)
        loss_mmd_partial_deriv = self._loss_mmd_deriv(z1, z3, z1_deriv, z3_deriv, alphas,
                                                      self.mmd_include_diag, self.mmd_sqrt)

        loss_mmd_total_deriv = loss_mmd_full_deriv + loss_mmd_partial_deriv

        loss_deriv = self.alpha * loss_moments_deriv + self.delta_sdp_corr * loss_sdp_corr_deriv \
            + self.gamma * loss_mmd_total_deriv + self.delta_corr * loss_corr_deriv
        return loss_deriv, loss_moments_deriv, loss_sdp_corr_deriv, loss_mmd_total_deriv, loss_corr_deriv

    @staticmethod
    def _loss_mmd(x, y, alphas, include_diag=False, sqrt=False):
        n = x.shape[0]
        m = y.shape[0]

        z = np.vstack((x, y))
        zzt = z @ z.T
        diag_zzt = zzt.diagonal()
        diag_zzt_mat = np.tile(diag_zzt, (n + m, 1))

        exponent = diag_zzt_mat + diag_zzt_mat.T - 2 * zzt
        loss = 0
        for alpha in alphas:
            loss_vals = np.exp(-exponent / (2 * alpha ** 2))
            if include_diag:
                loss += loss_vals[:n, :n].mean() + loss_vals[n:, n:].mean() - \
                    loss_vals[:n, n:].mean() - loss_vals[n:, :n].mean()
            else:
                np.fill_diagonal(loss_vals, 0.)
                loss += loss_vals[:n, :n].sum()/(n*(n-1.)) + loss_vals[n:, n:].sum()/(m*(m-1.)) - \
                    loss_vals[:n, n:].sum()/(n*m) - loss_vals[n:, :n].sum()/(n*m)
        # sqrt loss
        if sqrt:
            loss = np.sqrt(loss)
        return loss

    @staticmethod
    def _loss_mmd_deriv(x, y, x_deriv, y_deriv, alphas, include_diag=False, sqrt=False):
        n = x.shape[0]
        m = y.shape[0]
        n_pars = x_deriv.shape[2]

        z = np.vstack((x, y))
        zzt = z @ z.T
        diag_zzt = zzt.diagonal()
        diag_zzt_mat = np.tile(diag_zzt, (n + m, 1))

        exponent = diag_zzt_mat + diag_zzt_mat.T - 2 * zzt
        loss = 0
        loss_vals_for_deriv = np.zeros_like(exponent)
        for alpha in alphas:
            loss_vals = np.exp(-exponent / (2 * alpha ** 2))
            if include_diag:
                loss_vals_for_deriv += loss_vals * (-1/(2 * alpha ** 2))
                loss += loss_vals[:n, :n].mean() + loss_vals[n:, n:].mean() - \
                    loss_vals[:n, n:].mean() - loss_vals[n:, :n].mean()
            else:
                np.fill_diagonal(loss_vals, 0.)
                loss_vals_for_deriv += loss_vals * (-1/(2 * alpha ** 2))
                loss += loss_vals[:n, :n].sum()/(n*(n-1.)) + loss_vals[n:, n:].sum()/(m*(m-1.)) - \
                    loss_vals[:n, n:].sum()/(n*m) - loss_vals[n:, :n].sum()/(n*m)
        if sqrt:
            loss = np.sqrt(loss)

        loss_deriv = np.zeros(n_pars)
        for i_par in range(n_pars):
            z_deriv = np.vstack((x_deriv[:, :, i_par], y_deriv[:, :, i_par]))
            zz_deriv = z @ z_deriv.T
            diag_zz_deriv = zz_deriv.diagonal()
            diag_zz_deriv_mat = np.tile(diag_zz_deriv, (n + m, 1))

            exponent_deriv = (2 * diag_zz_deriv_mat + 2 * diag_zz_deriv_mat.T
                              - 2 * zz_deriv - 2 * zz_deriv.T)

            loss_vals_deriv = exponent_deriv * loss_vals_for_deriv
            if include_diag:
                loss_deriv[i_par] = loss_vals_deriv[:n, :n].mean() + loss_vals_deriv[n:, n:].mean() - \
                    loss_vals_deriv[:n, n:].mean() - loss_vals_deriv[n:, :n].mean()
            else:
                loss_deriv[i_par] = loss_vals_deriv[:n, :n].sum()/(n*(n-1.)) + loss_vals_deriv[n:, n:].sum()/(m*(m-1.)) - \
                    loss_vals_deriv[:n, n:].sum()/(n*m) - loss_vals_deriv[n:, :n].sum()/(n*m)

        if sqrt:
            loss_deriv = loss_deriv / (2 * loss)
        return loss_deriv


class KnockoffsDiagnostics:

    def compute_knock_diagnostics(self, x, x_knockoffs):
        n_obs = x.shape[0]
        dim_x = x.shape[1]
        n_obs_half = int(np.floor(n_obs / 2.))
        x_part1 = x[:n_obs_half, :]
        x_part2 = x[n_obs_half:, :]
        x_knockoffs_part1 = x_knockoffs[:n_obs_half, :]
        x_knockoffs_part2 = x_knockoffs[n_obs_half:, :]

        z1 = np.hstack((x_part1, x_knockoffs_part1))
        # full swap
        z2 = np.hstack((x_knockoffs_part2, x_part2))

        # partial swap
        swap_inds = np.arange(0, dim_x)[bernoulli.rvs(0.5, size=dim_x) == 1]
        z3 = np.hstack((x_part2, x_knockoffs_part2))
        z3[:, swap_inds] = x_knockoffs_part2[:, swap_inds]
        z3[:, swap_inds + dim_x] = x_part2[:, swap_inds]

        # cov
        cov_full = self.cov_distance(z1, z2)
        cov_partial = self.cov_distance(z1, z3)

        # MMD
        alphas = np.array([1., 2., 4., 8., 16., 32., 64., 128.])
        loss_mmd_full = KnockoffsLoss()._loss_mmd(z1, z2, alphas, include_diag=False, sqrt=False)
        loss_mmd_partial = KnockoffsLoss()._loss_mmd(z1, z3, alphas, include_diag=False, sqrt=False)

        # energy distance
        energy_full = self.energy_distance(z1, z2)
        energy_partial = self.energy_distance(z1, z3)

        # abs corr avg
        corr_mat = np.corrcoef(x, x_knockoffs, rowvar=False)
        abs_corr_avg = np.mean(np.diag(corr_mat, dim_x))

        diagnostics = {'abs_corr_avg': abs_corr_avg,
                       'full_swap': {
                           'cov': cov_full,
                           'mmd': loss_mmd_full,
                           'energy': energy_full
                       },
                       'partial_swap': {
                           'cov': cov_partial,
                           'mmd': loss_mmd_partial,
                           'energy': energy_partial
                       },
                       }

        return diagnostics

    @staticmethod
    def cov_distance(x, y):
        n = x.shape[0]
        m = y.shape[0]

        mu_x = x.mean(axis=0)
        mu_y = y.mean(axis=0)

        x_dm = x - mu_x
        y_dm = y - mu_y

        sigma_x = (x_dm @ x_dm.T) ** 2
        np.fill_diagonal(sigma_x, 0.)
        sigma_y = (y_dm @ y_dm.T) ** 2
        np.fill_diagonal(sigma_y, 0.)
        sigma_xy = (x_dm @ y_dm.T) ** 2

        phi_cov = sigma_x.sum() / (n * (n-1)) + sigma_y.sum() / (m * (m-1)) - \
            2. * sigma_xy.mean()

        return phi_cov

    @staticmethod
    def energy_distance(x, y):
        pw_dist_xx = cdist(x, x, metric='euclidean')
        pw_dist_yy = cdist(y, y, metric='euclidean')
        pw_dist_xy = cdist(x, y, metric='euclidean')

        energy_dist = 2 * pw_dist_xy.mean() - pw_dist_xx.mean() - pw_dist_yy.mean()

        return energy_dist
