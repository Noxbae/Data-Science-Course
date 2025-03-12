from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

image = imread("StillLife.jpg")
image_bw = np.mean(image, axis=-1)                              #Transform in black and white

U, S, Vt = np.linalg.svd(image_bw, full_matrices=False)
S = np.diag(S)

fig, ax = plt.subplots(1,4)
for i, r in enumerate([5, 20, 100]):
    reconstruct_image = U[:,:r] @ S[:r, :r] @ Vt[:r, :]
    ax[i].imshow(reconstruct_image, cmap="grey")
    ax[i].set_title(f"r = {r}")
    ax[i].axis("off")

ax[-1].imshow(image_bw, cmap="grey")
ax[-1].set_title("Original Image")
ax[-1].axis("off")
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(1,2)
ax[0].semilogy(np.diag(S))
ax[0].set_title("Singular values")
ax[1].plot(np.cumsum(np.diag(S)/np.sum(np.diag(S))))
ax[1].set_title("Cumulative Sum")
plt.tight_layout()
plt.show()