from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("external_data/icecream_sales.csv")
for col in df.columns:
    if df[col].dtype == "object":  # Check if the column is of object type (string)
        df[col] = df[col].map({"Yes": 1, "No": 0})

target = df["Ice Cream Sales ($,thousands)"]
features = df.drop(columns=["Ice Cream Sales ($,thousands)"])

model = LinearRegression()

reg = model.fit(features, target)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(features)

print(reg.score(features, target))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# # weights
# print(reg.coef_)
# # bias
# print(reg.intercept_)

if features.shape[1] == 2:
    ax.scatter(features.iloc[:, 0], features.iloc[:, 1], target)
    x_surf, y_surf = np.meshgrid(
        np.linspace(features.iloc[:, 0].min(), features.iloc[:, 0].max(), 100),
        np.linspace(features.iloc[:, 1].min(), features.iloc[:, 1].max(), 100),
    )

    # Calculate the predicted values for the meshgrid
    z_surf = reg.intercept_ + reg.coef_[0] * x_surf + reg.coef_[1] * y_surf

    # Plot the regression plane
    ax.plot_surface(
        x_surf, y_surf, z_surf, color="red", alpha=0.5, label="Regression Plane"
    )
    plt.show()
