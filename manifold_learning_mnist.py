from sklearn.manifold import Isomap, MDS, SpectralEmbedding, TSNE
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(X, y), _ = mnist.load_data()

# Randomly select n samples
n_samples = 1000
idx = np.random.choice(X.shape[0], n_samples, replace=False)
X, y = X[idx], y[idx]

# Reshape data to 2D arrays
X_2 = X.reshape(X.shape[0], -1)

# Reduce dimensionality using Isomap, MDS, Spectral Embedding, and t-SNE for n=2 and n=3
dimension_values = [2, 3]
dimension_reducers = [Isomap, MDS, SpectralEmbedding, TSNE]

X_reduced = []
for dim in dimension_values:
    for reducer in dimension_reducers:
        model = reducer(n_components=dim)
        X_reduced.append(model.fit_transform(X_2))

fig = plt.figure()
fig.suptitle('Manifold Learning Technique Comparision for MNIST')

# Plot 2D embeddings
X_2d = X_reduced[:4]
for i in range(4):
    ax = fig.add_subplot(2, 4, i+1)
    ax.scatter(X_2d[i][:, 0], X_2d[i][:, 1], c=y, cmap=plt.colormaps.get_cmap('jet'))
    ax.set_title(dimension_reducers[i].__name__ + ' 2D')

# Plot 3D embeddings
X_3d = X_reduced[4:]
for i in range(4):
    ax = fig.add_subplot(2, 4, i+5, projection='3d')
    sc = ax.scatter(X_3d[i][:, 0], X_3d[i][:, 1], X_3d[i][:, 2], c=y, cmap=plt.colormaps.get_cmap('jet'))
    ax.set_title(dimension_reducers[i].__name__ + ' 3D')

plt.legend(*sc.legend_elements(), loc='upper left', bbox_to_anchor=[-4.2, 1.7], fontsize=15)
plt.show()
