def estimate_price(mileage: int, theta0: float = 0, theta1: float = 0) -> float:
    return theta0 + theta1 * mileage
