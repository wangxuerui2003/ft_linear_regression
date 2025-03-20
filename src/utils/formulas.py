import numpy as np


def estimate_price(mileage, theta0: float, theta1: float):
    """
    The linear function for predicting price with mileage.
    mileage can be single value or a numpy array (vectorized).
    """
    return theta1 * mileage + theta0


def dJ_dw(x, y, w, b):
    """Partial derivative of weight against the MSE cost function (J)"""
    return np.mean((estimate_price(x, b, w) - y) * x)


def dJ_db(x, y, w, b):
    """Partial derivative of bias against the MSE cost function (J)"""
    return np.mean(estimate_price(x, b, w) - y)


def mse(x, y, w, b):
    return (1 / 2) * np.mean((estimate_price(x, b, w) - y) ** 2)


def r_squared(x, y, w, b):
    SS_res = np.sum((y - estimate_price(x, b, w)) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (SS_res / SS_tot)
