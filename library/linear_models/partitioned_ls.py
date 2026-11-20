r"""
Partitioned Least Squares

model: y = X_1\beta_1 + X_2\beta_2 + \epsilon
objective: minimize (y-X_1\beta_1-X_2\beta_2)'(y-X_1\beta_1-X_2\beta_2)

Notes:
1. Directly solve for P & M consumes a lot of memory
M1 is the projection matrix of X1 regression residual.
P1 = x1 @ np.linalg.inv(x1.T @ x1) @ x1.T
P2 = x2 @ np.linalg.inv(x2.T @ x2) @ x2.T
M1 = np.eye(P1.shape[0]) - P1
M2 = np.eye(P2.shape[0]) - P2
"""

import numpy as np
from library.utils import as_2d, residualize

class PLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, x1, x2, y):
        x1 = as_2d(x1)
        x2 = as_2d(x2)
        y = np.asarray(y, dtype=float)
        n = len(x1)

        if self.fit_intercept:
            x1 = np.column_stack((np.ones(n), x1))

        x1_tilde = residualize(x1, x2)
        y_tilde = residualize(y, x2)
        beta1 = np.linalg.solve(x1_tilde.T @ x1_tilde, x1_tilde.T @ y_tilde)

        x2_tilde = residualize(x2, x1)
        y_tilde = residualize(y, x1)
        beta2 = np.linalg.solve(x2_tilde.T @ x2_tilde, x2_tilde.T @ y_tilde)
        
        # e = y - x1 @ beta1 - x2 @ beta2

        self.coef1_ = beta1
        self.coef2_ = beta2

        self._X1 = x1
        self._X2 = x2
        self.beta1_ = beta1
        self.beta2_ = beta2

        return self

    def predict(self, x1, x2):
        x1 = as_2d(x1)
        x2 = as_2d(x2)
        n = len(x1)
        if self.fit_intercept:
            x1 = np.column_stack((np.ones(n), x1))

        return x1 @ self.coef1_ + x2 @ self.coef2_

