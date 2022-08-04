import numpy as np
try:
    import rpy2
except ImportError:
    ImportError('To estimate the margins with kde1d the python package rpy2 and the R package kde1d are required.')
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
r_kde1d_available = robjects.r('require("kde1d", quietly=TRUE)')[0]
if not r_kde1d_available:
    ImportError('To estimate the margins with kde1d the python package rpy2 and the R package kde1d are required.')


r_kde1d_fit = robjects.r('''
        kde_fit <- function(x) {
          return(kde1d::kde1d(x))
        }
        ''')

r_kde1d_discrete_fit = robjects.r('''
        kde_fit <- function(x, levels) {
          x <- ordered(x, levels=levels)
          return(kde1d::kde1d(x))
        }
        ''')

r_kde1d_cdf_eval = robjects.r('''
        cdf_eval <- function(kde_fit, x) {
          return(kde1d::pkde1d(x, kde_fit))
        }
        ''')

r_kde1d_invcdf_eval = robjects.r('''
        invcdf_eval <- function(kde_fit, x) {
          return(kde1d::qkde1d(x, kde_fit))
        }
        ''')

r_kde1d_pdf_eval = robjects.r('''
        pdf_eval <- function(kde_fit, x) {
          return(kde1d::dkde1d(x, kde_fit))
        }
        ''')


class KDE1D:

    def __init__(self):
        self._kdefit = None
        self._discrete_levels = None

    def fit(self, x, discrete=False):
        if not discrete:
            self._kdefit = r_kde1d_fit(x)
        else:
            self._discrete_levels = np.unique(x)
            self._kdefit = r_kde1d_discrete_fit(x, self._discrete_levels)
        return self

    def ppf(self, x):
        return r_kde1d_invcdf_eval(self._kdefit, x)

    def cdf(self, x):
        if self._discrete_levels is None:
            res = r_kde1d_cdf_eval(self._kdefit, x)
        else:
            # bring the x values to the next smaller value in self._discrete_levels
            ind = np.searchsorted(self._discrete_levels, x, side='right')
            res = r_kde1d_cdf_eval(self._kdefit, self._discrete_levels[ind - 1])
            res[ind == 0] = 0.
        return res

    def pdf(self, x):
        return r_kde1d_pdf_eval(self._kdefit, x)
