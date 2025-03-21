import matplotlib.pyplot as plt
import numpy as np
from utils.load_data import load_data_csv


def fit_plot(
    w,
    b,
    dataset_path: str = "data.csv",
    x_col_name: str = "km",
    y_col_name: str = "price",
):
    data = load_data_csv(dataset_path)

    x = data[x_col_name]
    y = data[y_col_name]

    _, ax = plt.subplots()
    ax.set_xlabel(x_col_name)
    ax.set_ylabel(y_col_name)
    ax.set_title(f"{x_col_name} to {y_col_name} Linear Regression")
    ax.scatter(x, y)

    # line = np.linspace(x.min(), x.max(), 100)
    # ax.plot(line, w * line + b, linestyle="solid")

    sorted_x = np.sort(x)
    ax.plot(sorted_x, w * sorted_x + b, color="red")

    plt.show()
