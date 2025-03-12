import numpy as np
import matplotlib.pyplot as plt

#Generate data
m = -4                                                          #Generate true coefficient
x = np.linspace(-2, 2, 20)                                      #Define x
y = m*x + np.random.randn(x.size)                               #Add gaussian noice

#Start ploting
plt.plot(x, x*m, label="True line", color="black")
plt.scatter(x, y, label="Noisy data", color="red")

x = x.reshape(-1, 1)

U, S, Vt = np.linalg.svd(x, full_matrices=False)
mtilde = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y             #Remember that S is an array so we need to reshape it
plt.plot(x, mtilde*x, "--", label="Regression line")
plt.legend()
plt.show()