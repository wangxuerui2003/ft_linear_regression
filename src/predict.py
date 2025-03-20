import numpy as np
import sys
from utils.formulas import estimate_price

PARAMS_FILEPATH = "params.txt"

try:
    params = np.loadtxt(PARAMS_FILEPATH, dtype=float)
except FileNotFoundError:
    print("Please train the params first!", file=sys.stderr)
    exit(1)

theta0 = params[0]
theta1 = params[1]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <mileage>", file=sys.stderr)
        exit(1)
    try:
        input = float(sys.argv[1])
    except ValueError:
        print("Invalid mileage value.", file=sys.stderr)
        exit(1)
    print("Predicted price:", round(estimate_price(input, theta0, theta1)))
