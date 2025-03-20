import numpy as np
import sys
from utils.load_data import load_data_csv
from utils.formulas import r_squared

df = load_data_csv()
mileages = df["km"].to_numpy()
prices = df["price"].to_numpy()

try:
    params = np.loadtxt("params.txt", dtype=float)
except FileNotFoundError:
    print("Please train the params first!", file=sys.stderr)
    exit(1)

theta0 = params[0]
theta1 = params[1]


if __name__ == "__main__":
    print("Model accuracy:", r_squared(mileages, prices, theta1, theta0))
