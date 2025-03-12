import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing_data, housing_value = fetch_california_housing(return_X_y=True)
labels = fetch_california_housing().feature_names

print(housing_data.shape, housing_value.shape)
print(labels)

plt.scatter(housing_data[:, 0], housing_value)
plt.xlabel("Median income x1e4")
plt.ylabel("Median House value x1e5")
plt.show()
plt.close()

housing_data = (housing_data - np.mean(housing_data, axis=0)) / np.std(housing_data, axis=0)
housing_data = np.pad(housing_data, ((0,0), (0,1)), mode="constant", constant_values=1)

U, S, Vt = np.linalg.svd(housing_data, full_matrices=False)

x = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ housing_value

plt.scatter(housing_value, housing_data @ x, s=1)
plt.plot([0,6], [0, 6], "k--")
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.show()
plt.close()

plt.bar(range(1, len(x)), x[:-1], width=0.5)
plt.xlabel("Feature")
plt.ylabel("Influence")
plt.xticks(range(1, len(x)), labels, rotation=45)
plt.show()