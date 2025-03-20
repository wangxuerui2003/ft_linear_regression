import matplotlib.pyplot as plt
import numpy as np
from utils.load_data import load_data_csv


def fit_plot(w, b):
    data = load_data_csv()

    x = data["km"]
    y = data["price"]

    fig, ax = plt.subplots()
    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel("Price")
    ax.set_title("Mileage to Price Scatter plot")
    ax.scatter(x, y)

    # line = np.linspace(x.min(), x.max(), 100)
    # ax.plot(line, w * line + b, linestyle="solid")

    sorted_x = np.sort(x)
    ax.plot(sorted_x, w * sorted_x + b, color="red")

    plt.show()
