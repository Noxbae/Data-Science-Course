import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.chdir("ESE-04-Unsupervised machine learning")
metadata = json.load(open("dataset/info.json"))
Nx, Ny = metadata["global"]["Nxyz"]

X = np.fromfile("dataset/grid/X_m.dat", dtype="<f4").reshape(Ny, Nx)
Y = np.fromfile("dataset/grid/Y_m.dat", dtype="<f4").reshape(Ny, Nx)

T = np.fromfile("dataset/data/T_K_id0100.dat", dtype="<f4").reshape(Nx, Ny).T
YOH = np.fromfile("dataset/data/YOH_id0100.dat", dtype="<f4").reshape(Nx, Ny).T
YH2 = np.fromfile("dataset/data/YH2_id0100.dat", dtype="<f4").reshape(Nx, Ny).T

fig, ax = plt.subplots(1, 3)
# Plot Temperature
ax[0].set_title("Temperature")
ax[0].pcolormesh(Y, X, T, cmap="inferno")

# Plot YOH concentration
ax[1].set_title("YOH")
ax[1].pcolormesh(Y, X, YOH, cmap="inferno")

# Plot YH2 concentration
ax[2].set_title("YH2")
ax[2].pcolormesh(Y, X, YH2, cmap="inferno")
plt.tight_layout()
plt.show()

features = np.array([])
for filename in os.listdir("dataset/data"):
    if filename.endswith(".dat"):
        if features.size == 0:
            features = np.fromfile(f"dataset/data/{filename}", dtype="<f4")
        else:
            features = np.vstack((features, np.fromfile(f"dataset/data/{filename}", dtype="<f4")))

features = features.T

c = np.mean(features, axis=0)
d = np.std(features, axis=0)
features = (features- c)/d

from sklearn.cluster import KMeans
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(features)
colors = kmeans.labels_.reshape(Nx, Ny).T

#rng = np.random.RandomState(0)
#index = rng.permutation(len(features))[:100000]

plt.pcolormesh(Y, X, colors, cmap=plt.get_cmap("viridis", n_clusters))
plt.colorbar()
plt.title("K-means Clustering")
plt.show()

#plt.scatter(YOH.flatten()[index], T.flatten()[index],c=colors.flatten()[index], cmap='viridis', s=1)
plt.scatter(YOH.flatten(), T.flatten(),c=colors.flatten(), cmap='viridis', s=1)
plt.xlabel("YOH")
plt.ylabel("Temperature (K)")
plt.title("K-means Clustering in Temperature vs YOH Space")
plt.show()