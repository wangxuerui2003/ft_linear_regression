import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.load_data import load_data_csv
from utils.visualization import fit_plot
from utils.formulas import r_squared, dJ_dw, dJ_db, mse


PARAMS_FILEPATH = "params.txt"

visual = False
verbose = False

# learning rate
eta = 0.5

# training iterations
max_epochs = 100

# if next iteration changes weights and bias less than this value then stop traning
epsilon = 0.0001

# weight/slope/theta1
w = 0

# bias/y-intercept/theta0
b = 0

# dataset
df = load_data_csv()

# numpy array of features and targets
mileages_orig = df["km"].to_numpy()
prices_orig = df["price"].to_numpy()

# normalization
mileages = (mileages_orig - np.mean(mileages_orig)) / np.std(mileages_orig)
prices = (prices_orig - np.mean(prices_orig)) / np.std(prices_orig)

# number of observations
m = df.shape[0]


def train():
    global w, b

    if visual:
        plt.figure(figsize=(8, 6))
        plt.scatter(mileages_orig, prices_orig, label="Data Points")

    for e in range(max_epochs):
        if verbose:
            print(f"epoch: {e + 1}, loss (mse): {mse(mileages, prices, w, b)}")

        # gradient descent
        dw = dJ_dw(mileages, prices, w, b)
        db = dJ_db(mileages, prices, w, b)
        w -= eta * dw
        b -= eta * db

        # early stopping
        if abs(dw) < epsilon and abs(db) < epsilon:
            print("Early stopped at epoch", e + 1)
            break

        if visual:
            w_orig, b_orig = denormalize_params()
            plt.plot(
                mileages_orig,
                w_orig * mileages_orig + b_orig,
                color="red",
                label=f"Epoch {e + 1}",
            )
            plt.title("Gradient Descent Visualization")
            plt.xlabel("Mileage")
            plt.ylabel("Price")
            plt.legend()
            plt.pause(0.1)
            # Clear the previous line (except for the first epoch)
            if e < max_epochs - 1:
                plt.gca().lines[-1].remove()

    if visual:
        plt.show()


def denormalize_params():
    # de-normalize w and b
    w_orig = w * np.std(prices_orig) / np.std(mileages_orig)
    b_orig = (
        b * np.std(prices_orig) - np.mean(mileages_orig) * w_orig + np.mean(prices_orig)
    )
    return w_orig, b_orig


def save_params(theta0, theta1):
    params = np.array([theta0, theta1])
    np.savetxt(PARAMS_FILEPATH, params, fmt="%f")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--visual",
        action="store_true",
        required=False,
        help="Turn on gradient descent visualization",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="Logs info on each epoch.",
    )
    args = vars(ap.parse_args())
    if args["visual"]:
        global visual
        visual = True
    if args["verbose"]:
        global verbose
        verbose = True


if __name__ == "__main__":
    parse_args()

    train()
    print(f"Accuracy (R^2): {r_squared(mileages, prices, w, b)}")

    # save theta0 and theta1
    w_orig, b_orig = denormalize_params()
    save_params(b_orig, w_orig)

    if not visual:
        fit_plot(w_orig, b_orig)
