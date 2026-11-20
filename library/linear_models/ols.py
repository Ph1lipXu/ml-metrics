r"""
Ordinary Least Squares

model: y = X\beta + \epsilon
objective: minimize (y-X\beta)'(y-X\beta)
solution: \beta = (X'X)^{-1}X'y
"""

import numpy as np
from library.utils import as_2d

class OLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, x, y):
        x = as_2d(x)
        y = np.asarray(y, dtype=float)

        n = x.shape[0]
        if self.fit_intercept:
            X = np.column_stack((np.ones(n), x))
        else:
            X = x

        beta = np.linalg.solve(X.T @ X, X.T @ y)
        e = y - X @ beta
        s2 = e.T @ e / (n - X.shape[1])
        var_beta = s2 * np.linalg.inv(X.T @ X)
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self._X = X
        self.beta_ = beta
        self.sigma2_ = s2
        self.cov_ = var_beta
        self.se_ = np.sqrt(np.diag(var_beta))
        return self

    def predict(self, x):
        x = as_2d(x)
        return x @ self.coef_ + self.intercept_
