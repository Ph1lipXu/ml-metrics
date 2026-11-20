import numpy as np


def residualize(a, x):
    coef = np.linalg.solve(x.T @ x, x.T @ a)
    return a - x @ coef