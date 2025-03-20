import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

np.random.seed(0)
A = np.random.randn (100,10)
x = np.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 0])
b = A @ x + np.random.randn(100)

xL2 = np.linalg.pinv(A) @ b

reg = linear_model.Lasso(alpha=0.2).fit(A, b)
xLasso = reg.coef_

width = 0.2
positions = np.arange(len(x))

plt.bar(positions-width, x, width=width, label="True x")
plt.bar(positions, xL2, width=width, label="Pinv")
plt.bar(positions+width, xLasso, width=width, label="Lasso")
plt.legend()
plt.show()