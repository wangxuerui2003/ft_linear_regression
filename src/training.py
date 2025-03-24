import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.load_data import load_data_csv
from utils.visualization import visualize_regression
from utils.formulas import r_squared, dJ_dw, dJ_db, mse


PARAMS_FILEPATH = "params.txt"

dataset_path = "data.csv"

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


def load_dataset():
    global df, x_orig, y_orig, x, y, ws, b, feature_names, target_name
    # dataset
    df = load_data_csv(dataset_path)

    for col in df.columns:
        if df[col].dtype == "object":  # Check if the column is of object type (string)
            df[col] = df[col].map({"Yes": 1, "No": 0})

    feature_names = [col for col in df.columns if col != y_col_name]
    target_name = y_col_name

    # numpy array of features and targets
    y_orig = df[y_col_name].to_numpy()
    x_orig = df.drop(columns=[y_col_name]).to_numpy()

    # normalization
    x = x_orig.copy().astype("float64")
    for i in range(x_orig.shape[1]):
        x[:, i] = (x_orig[:, i] - np.mean(x_orig[:, i])) / np.std(x_orig[:, i])
    y = (y_orig - np.mean(y_orig)) / np.std(y_orig)

    # weight/slope/theta1
    ws = np.zeros(x.shape[1])

    # bias/y-intercept/theta0
    b = 0


def train():
    global ws, b

    # if visual:
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(x_orig, y_orig, label="Data Points")

    for e in range(max_epochs):
        if verbose:
            print(f"epoch: {e + 1}, loss (mse): {mse(x, y, ws, b)}")

        # gradient descent
        dw = dJ_dw(x, y, ws, b)
        db = dJ_db(x, y, ws, b)
        ws -= eta * dw
        b -= eta * db

        # early stopping
        if early_stop and (abs(max(dw)) < epsilon and abs(db) < epsilon):
            print("Early stopped at epoch", e + 1)
            break

    if visual:
        plt.show()


def denormalize_params():
    # de-normalize w and b
    w_orig = ws * np.std(y_orig) / np.std(x_orig, axis=0)
    b_orig = b * np.std(y_orig) - np.mean(x_orig, axis=0) @ w_orig + np.mean(y_orig)
    return w_orig, b_orig


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
    print(f"Accuracy (R^2): {r_squared(x, y, ws, b)}")

    w_orig, b_orig = denormalize_params()
    print("ws:", w_orig)
    print("b:", b_orig)

    if visual and len(ws) <= 2:
        visualize_regression(x, y, ws, b, feature_names, target_name)
