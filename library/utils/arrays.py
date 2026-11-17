import numpy as np


def as_2d(x):
    """Return x as a float array of shape (n, k). A 1-D array becomes one column (n, 1)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x