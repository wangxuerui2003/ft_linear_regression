import matplotlib.pyplot as plt
import numpy as np
from utils.load_data import load_data_csv


def visualize_regression(xs, y, ws, b, feature_names, target_name):
    num_features = xs.shape[1]

    if num_features == 1:
        # 2D Plot
        plt.figure()
        plt.scatter(xs, y, label="Data Points")

        plt.plot(
            xs,
            ws * xs + b,
            color="red",
        )
        plt.title("Linear Regression Visualization")
        plt.xlabel(feature_names[0])
        plt.ylabel(target_name)
        plt.legend()
        plt.pause(0.1)

        plt.show()

    elif num_features == 2:
        # 3D Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], y, label="Data Points")

        x_surf, y_surf = np.meshgrid(
            np.linspace(xs[:, 0].min(), xs[:, 0].max(), 100),
            np.linspace(xs[:, 1].min(), xs[:, 1].max(), 100),
        )

        z_surf = ws[0] * x_surf + ws[1] * y_surf + b
        ax.plot_surface(
            x_surf,
            y_surf,
            z_surf,
            color="red",
            alpha=0.5,
        )

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(target_name)
        ax.set_title("3D Linear Regression Visualization")
        ax.legend()
        plt.pause(0.1)

        plt.show()
    else:
        print("Visualization only supports 2D or 3D data.")
