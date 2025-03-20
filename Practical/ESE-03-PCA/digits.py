from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

digits_images = load_digits().data
digits_label = load_digits().target
#print(digits_images.shape)

'''
index = 0
print(digits_label[index])
digit = digits_images[index, :].reshape(8, 8)
plt.imshow(digit, cmap="Greys")
plt.show()
'''
pca = PCA()
pca.fit(digits_images)

principal_comp = pca.components_.T

#fig, ax = plt.subplots(1, 2)
#ax[0].imshow(principal_comp[:, 0].reshape(8, 8), cmap="Greys")
#ax[1].imshow(principal_comp[:, 1].reshape(8, 8), cmap="Greys")
#plt.show()

pca_scores = digits_images @ principal_comp

plt.scatter(pca_scores[:, 0], pca_scores[:, 1],
            c=digits_label,
            alpha=0.5,
            s=10,
            cmap=plt.cm.get_cmap("jet",10))
plt.colorbar()
plt.show()