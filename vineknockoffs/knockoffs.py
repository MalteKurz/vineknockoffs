import numpy as np
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
        # loss_mmd_full = fast_loss_mmd(z1, z2, alphas)
        # partial swap
        if swap_inds is not None:
            z3 = np.hstack((x_part2, x_knockoffs_part2))
            z3[:, swap_inds] = x_knockoffs_part2[:, swap_inds]
            z3[:, swap_inds + dim_x] = x_part2[:, swap_inds]
            loss_mmd_partial = self._loss_mmd(z1, z3, alphas)
        else:
            loss_mmd_partial = 0.
        # loss_mmd_partial = fast_loss_mmd(z1, z3, alphas)

        loss_mmd_total = loss_mmd_full + loss_mmd_partial

        loss = self.alpha * loss_moments + self.delta_sdp_corr * loss_sdp_corr \
            + self.gamma * loss_mmd_total + self.delta_corr * loss_corr
        return loss, loss_moments, loss_sdp_corr, loss_mmd_total, loss_corr

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
