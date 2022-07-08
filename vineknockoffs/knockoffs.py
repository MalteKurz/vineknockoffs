import numpy as np
from scipy.stats import bernoulli

from ._utils_gaussian_knockoffs import sdp_solver


class KockoffsLoss:

    def __init__(self, alpha=1., delta_sdp_corr=1., gamma=1., delta_corr=0.):
        self._alpha = alpha
        self._delta_sdp_corr = delta_sdp_corr
        self._gamma = gamma
        self._delta_corr = delta_corr

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
        loss_mmd_full = self._loss_mmd(z1, z2, alphas)

        # partial swap
        z3 = np.hstack((x_part2, x_knockoffs_part2))
        z3[:, swap_inds] = x_knockoffs_part2[:, swap_inds]
        z3[:, swap_inds + dim_x] = x_part2[:, swap_inds]
        loss_mmd_partial = self._loss_mmd(z1, z3, alphas)

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

        loss_mmd_full_deriv = self._loss_mmd_deriv(z1, z2, z1_deriv, z2_deriv, alphas)
        loss_mmd_partial_deriv = self._loss_mmd_deriv(z1, z3, z1_deriv, z3_deriv, alphas)

        loss_mmd_total_deriv = loss_mmd_full_deriv + loss_mmd_partial_deriv

        loss_deriv = self.alpha * loss_moments_deriv + self.delta_sdp_corr * loss_sdp_corr_deriv \
            + self.gamma * loss_mmd_total_deriv + self.delta_corr * loss_corr_deriv
        return loss_deriv, loss_moments_deriv, loss_sdp_corr_deriv, loss_mmd_total_deriv, loss_corr_deriv

    @staticmethod
    def _loss_mmd(x, y, alphas):
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
            loss += loss_vals[:n, :n].mean() + loss_vals[n:, n:].mean() - \
                loss_vals[:n, n:].mean() - loss_vals[n:, 1:n].mean()
        # sqrt loss
        loss = np.sqrt(loss)
        return loss

    @staticmethod
    def _loss_mmd_deriv(x, y, x_deriv, y_deriv, alphas):
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
            loss_vals_for_deriv += loss_vals * (-1/(2 * alpha ** 2))
            loss += loss_vals[:n, :n].mean() + loss_vals[n:, n:].mean() - \
                loss_vals[:n, n:].mean() - loss_vals[n:, 1:n].mean()
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
            loss_deriv[i_par] = loss_vals_deriv[:n, :n].mean() + loss_vals_deriv[n:, n:].mean() - \
                loss_vals_deriv[:n, n:].mean() - loss_vals_deriv[n:, 1:n].mean()

        loss_deriv = loss_deriv / (2 * loss)
        return loss_deriv
