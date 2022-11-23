Introduction
============

Simulate from a Gaussian copula and use mixed-normal distributions as margins

.. ipython:: python

    import numpy as np
    from scipy.stats import multivariate_normal, norm, invwishart
    from scipy.linalg import toeplitz
    from scipy.optimize import root_scalar

    class GaussianMixture:
        def __init__(self, weights, mus, sigmas):
            self.weights = weights
            self.mus = mus
            self.sigmas = sigmas
            self.n_components = len(weights)
        def cdf(self, x):
            res = np.zeros_like(x)
            for i_component in range(self.n_components):
                res += self.weights[i_component] * norm.cdf(x,
                                                            loc=self.mus[i_component],
                                                            scale=self.sigmas[i_component])
            return res
        def ppf(self, x):
            bracket = [-10., 10.]
            if x.min() < self.cdf(bracket[0:1]):
                bracket[0] -= 50.
            if x.max() > self.cdf(bracket[1:2]):
                bracket[1] += 50.
            res = np.array([root_scalar(lambda yy: self.cdf(np.array([[yy]]).T) - x[i],
                                        bracket=bracket,
                                        method='brentq',
                                        xtol=1e-12, rtol=1e-12).root for i in range(len(x))])
            return res

.. ipython:: python

    n_vars = 5
    margins = [None]*n_vars
    for i_var in range(n_vars):
        n_components = 3
        weights = np.full(n_components, 1./n_components)
        mus = norm.rvs(size=n_components, scale=2.)
        sigmas = np.sqrt(invwishart.rvs(df=3, scale=1.0, size=n_components))
        margins[i_var] = GaussianMixture(weights, mus, sigmas)

.. ipython:: python

    cov_mat = toeplitz([np.power(0.5, k) for k in range(5)])
    u_train = norm.cdf(multivariate_normal(mean=np.zeros(5), cov=cov_mat).rvs(1000))
    x_train = np.full_like(u_train, np.nan)
    for i_var in range(n_vars):
        x_train[:, i_var] = margins[i_var].ppf(u_train[:, i_var])

.. ipython:: python

    u_test = norm.cdf(multivariate_normal(mean=np.zeros(5), cov=cov_mat).rvs(1000))
    x_test = np.full_like(u_test, np.nan)
    for i_var in range(n_vars):
        x_test[:, i_var] = margins[i_var].ppf(u_test[:, i_var])

Gaussian knockoffs
------------------

Estimate a Gaussian knockoff model

.. ipython:: python

    from vineknockoffs.vine_knockoffs import VineKnockoffs
    gau_ko = VineKnockoffs()
    gau_ko.fit_gaussian_knockoffs(x_train=x_train)
    gau_ko._dvine.copulas

Generate a knockoff copy

.. ipython:: python

    x_gau_ko = gau_ko.generate(x_test)

Pairwise scatter plots

.. ipython:: python

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(font_scale=1)
    @savefig gau_ko.png width=10in
    sns.pairplot(pd.DataFrame(np.hstack((x_test, x_gau_ko))))

Estimated correlations

.. ipython:: python

    sns.set(font_scale=3)
    @savefig gau_ko_corr.png width=7in height=5in
    sns.heatmap(np.corrcoef(np.hstack((x_test, x_gau_ko)).T), annot=True);

Gaussian copula knockoffs
-------------------------

Estimate a Gaussian copula knockoff model

.. ipython:: python

    from vineknockoffs.vine_knockoffs import VineKnockoffs
    gau_cop_ko = VineKnockoffs()
    gau_cop_ko.fit_gaussian_copula_knockoffs(x_train=x_train)
    gau_cop_ko._dvine.copulas

Generate a knockoff copy

.. ipython:: python

    x_gau_cop_ko = gau_cop_ko.generate(x_test)

Pairwise scatter plots

.. ipython:: python

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(font_scale=1)
    @savefig gau_cop_ko.png width=10in
    sns.pairplot(pd.DataFrame(np.hstack((x_test, x_gau_cop_ko))))

Estimated correlations

.. ipython:: python

    sns.set(font_scale=3)
    @savefig gau_cop_ko_corr.png width=7in height=5in
    sns.heatmap(np.corrcoef(np.hstack((x_test, x_gau_cop_ko)).T), annot=True);

Vine copula knockoffs
---------------------

Estimate a Gaussian copula knockoff model

.. ipython:: python
    :okwarning:

    from vineknockoffs.vine_knockoffs import VineKnockoffs
    vine_ko = VineKnockoffs()
    vine_ko.fit_vine_copula_knockoffs(x_train=x_train)
    vine_ko._dvine.copulas

Generate a knockoff copy

.. ipython:: python

    x_vine_ko = vine_ko.generate(x_test)

Pairwise scatter plots

.. ipython:: python

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(font_scale=1)
    @savefig vine_ko.png width=10in
    sns.pairplot(pd.DataFrame(np.hstack((x_test, x_vine_ko))))

Estimated correlations

.. ipython:: python

    sns.set(font_scale=3)
    @savefig vine_ko_corr.png width=7in height=5in
    sns.heatmap(np.corrcoef(np.hstack((x_test, x_vine_ko)).T), annot=True);
