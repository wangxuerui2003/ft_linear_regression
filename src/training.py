import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from utils.load_data import load_data_csv
from utils.visualization import fit_plot
from utils.formulas import r_squared, dJ_dw, dJ_db, mse


PARAMS_FILEPATH = "params.txt"

dataset_path = "data.csv"

x_col_name = "km"
y_col_name = "price"

# cli variables (optional)
visual = False
verbose = False

# learning rate
eta = 0.5

# training iterations
max_epochs = 100

# early stopping
early_stop = False
# dw and db smaller than epsilon then early stop (if early stopping on)
epsilon = 0.0001

# weight/slope/theta1
w = 0

# bias/y-intercept/theta0
b = 0


def load_dataset():
    global df, x_orig, y_orig, x, y
    # dataset
    df = load_data_csv(dataset_path)

    # numpy array of features and targets
    x_orig = df[x_col_name].to_numpy()
    y_orig = df[y_col_name].to_numpy()

    # normalization
    x = (x_orig - np.mean(x_orig)) / np.std(x_orig)
    y = (y_orig - np.mean(y_orig)) / np.std(y_orig)


def train():
    global w, b

    if visual:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_orig, y_orig, label="Data Points")

    for e in range(max_epochs):
        if verbose:
            print(f"epoch: {e + 1}, loss (mse): {mse(x, y, w, b)}")

        # gradient descent
        dw = dJ_dw(x, y, w, b)
        db = dJ_db(x, y, w, b)
        w -= eta * dw
        b -= eta * db

        # early stopping
        if early_stop and (abs(dw) < epsilon and abs(db) < epsilon):
            print("Early stopped at epoch", e + 1)
            break

        if visual:
            w_orig, b_orig = denormalize_params()
            plt.plot(
                x_orig,
                w_orig * x_orig + b_orig,
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
    w_orig = w * np.std(y_orig) / np.std(x_orig)
    b_orig = b * np.std(y_orig) - np.mean(x_orig) * w_orig + np.mean(y_orig)
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
        help="Logs info on each epoch",
    )
    ap.add_argument(
        "--early-stop",
        action="store_true",
        required=False,
        help="Early stop when dw or db smaller than epsilon",
    )
    ap.add_argument(
        "--lr",
        required=False,
        help="Custom learning rate",
    )
    ap.add_argument(
        "--epochs",
        required=False,
        help="Custom max epochs",
    )
    ap.add_argument(
        "--dataset-path",
        required=False,
        help="Custom dataset x column name",
    )
    ap.add_argument(
        "-x",
        "--x-col-name",
        required=False,
        help="Custom dataset x column name",
    )
    ap.add_argument(
        "-y",
        "--y-col-name",
        required=False,
        help="Custom dataset y column name",
    )
    args = vars(ap.parse_args())
    if args["visual"]:
        global visual
        visual = True
    if args["verbose"]:
        global verbose
        verbose = True
    if args["early_stop"]:
        global early_stop
        early_stop = True
    if args["dataset_path"]:
        global dataset_path
        dataset_path = args["dataset_path"]
    if args["x_col_name"]:
        global x_col_name
        x_col_name = args["x_col_name"]
    if args["y_col_name"]:
        global y_col_name
        y_col_name = args["y_col_name"]
    if args["lr"]:
        global eta
        try:
            eta = float(args["lr"])
        except ValueError:
            print("Invalid learning rate, must be a float.")
            exit(1)
    if args["epochs"]:
        global max_epochs
        try:
            max_epochs = int(args["epochs"])
            if max_epochs <= 0:
                raise ValueError
        except ValueError:
            print("Invalid epochs, must be a positive int.")
            exit(1)


if __name__ == "__main__":
    parse_args()
    load_dataset()

    train()
    print(f"Accuracy (R^2): {r_squared(x, y, w, b)}")

    # save theta0 and theta1
    w_orig, b_orig = denormalize_params()
    save_params(b_orig, w_orig)

    if not visual:
        fit_plot(w_orig, b_orig, dataset_path, x_col_name, y_col_name)
