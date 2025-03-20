import numpy as np
import matplotlib.pyplot as plt

xC = np.array([2, 1])
sig = np.array([2, 0.5])
theta = np.pi/3

n_points = 10000

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

X = R @ np.diag(sig) @ np.random.randn(2, n_points) + np.diag(xC) @ np.ones((2, n_points))

plt.scatter(X[0,:], X[1, :], color="k", s=1)

Xavg = np.mean(X, axis=1)
#B = X - Xavg.reshape(-1, 1)

#B = B/np.sqrt(n_points)
B = X - Xavg[:, np.newaxis]

U, S, Vt = np.linalg.svd(B/np.sqrt(n_points), full_matrices=False)

plt.plot(np.array([Xavg[0], Xavg[0]+U[0, 0]*S[0]]),
         np.array([Xavg[1], Xavg[1]+U[1, 0]*S[0]]), '-', color='cyan', linewidth=3)

plt.plot(np.array([Xavg[0], Xavg[0]+U[0, 1]*S[1]]),
         np.array([Xavg[1], Xavg[1]+U[1, 1]*S[1]]), '-', color='cyan', linewidth=3)
plt.axis("equal")
plt.show()