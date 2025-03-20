from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
x = df["km"].to_numpy().reshape(-1, 1)
y = df["price"].to_numpy()

model = LinearRegression()

reg = model.fit(x, y)
print(reg.score(x, y))

y_pred = reg.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color="red")
plt.show()
