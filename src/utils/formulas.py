import numpy as np


def predict(xs, w, b):
    """
    Features can be 1d array (single feature) or 2d array (multi features).
    Weights (w) can be scalar or an array of weights.
    bias (b) is scalar.
    """
    return xs @ w + b


def dJ_dw(xs, y, w, b, i: int | None = None):
    """Partial derivative of weight (Wi) against the MSE cost function (J)"""
    return np.mean((predict(xs, w, b) - y)[:, np.newaxis] * xs, axis=0)


def dJ_db(xs, y, w, b):
    """Partial derivative of bias against the MSE cost function (J)"""
    return np.mean(predict(xs, w, b) - y)


def mse(xs, y, w, b):
    return (1 / 2) * np.mean((predict(xs, w, b) - y) ** 2)


def r_squared(xs, y, w, b):
    SS_res = np.sum((y - predict(xs, w, b)) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (SS_res / SS_tot)
