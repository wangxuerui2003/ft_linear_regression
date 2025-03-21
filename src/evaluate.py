import numpy as np
import sys
import argparse
from utils.load_data import load_data_csv
from utils.formulas import r_squared


dataset_path = "data.csv"
x_col_name = "km"
y_col_name = "price"


def evaluate_accuracy():
    df = load_data_csv(dataset_path)
    x = df[x_col_name].to_numpy()
    y = df[y_col_name].to_numpy()

    try:
        params = np.loadtxt("params.txt", dtype=float)
    except FileNotFoundError:
        print("Please train the params first!", file=sys.stderr)
        exit(1)

    theta0 = params[0]
    theta1 = params[1]

    print("Model accuracy:", r_squared(x, y, theta1, theta0))


def parse_args():
    ap = argparse.ArgumentParser()
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
    if args["dataset_path"]:
        global dataset_path
        dataset_path = args["dataset_path"]
    if args["x_col_name"]:
        global x_col_name
        x_col_name = args["x_col_name"]
    if args["y_col_name"]:
        global y_col_name
        y_col_name = args["y_col_name"]


if __name__ == "__main__":
    parse_args()
    evaluate_accuracy()
