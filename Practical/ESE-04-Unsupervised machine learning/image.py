import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

os.chdir("ESE-04-Unsupervised machine learning")
image = imread("StillLife.jpg")
pixels = image.reshape(-1, image.shape[2])

def plot_colors(data, title, colors=None, samples=10000):
    if colors is None:
        colors=data
    rng = np.random.RandomState(0)
    index = rng.permutation(len(data))[:samples]

    R, G, B = data[index].T
    norm_colors = colors[index]/255

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(R, B, c=norm_colors, marker=".")
    ax[0].set(xlabel="Red", ylabel="Blue", xlim=(0, 255), ylim=(0, 255))

    ax[1].scatter(R, G, c=norm_colors, marker=".")
    ax[1].set(xlabel="Red", ylabel="Green", xlim=(0, 255), ylim=(0, 255))
    fig.suptitle(title, size=20)
    plt.show()

plot_colors(pixels, title="Original 16mln colors")

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=16)
kmeans.fit(pixels)
reduced_colors = kmeans.cluster_centers_[kmeans.predict(pixels)].astype(int)

plot_colors(pixels, colors=reduced_colors, title="Reduced Colors")

image_recolored = reduced_colors.reshape(image.shape)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(image_recolored)
ax[1].set_title('16-color Image')
plt.show()