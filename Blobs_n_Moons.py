import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_moons
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=200, centers=5, cluster_std=1.5, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap="viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
plt.show()

'''
---------------------------------------------------------------------------------------------------------------
'''
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(X[:,0], X[:,1])
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.labels_

plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap="viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
plt.show()