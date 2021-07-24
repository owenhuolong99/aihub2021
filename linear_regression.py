from sklearn.linear_model import LinearRegression
import numpy as np

x = np.linspace(0, 1)
y = 3 * x + 4 + np.random.rand() / 100
x = x.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)