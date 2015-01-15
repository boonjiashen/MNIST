"""Learn a dictionary from MNIST digits
"""

import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import util
import time
import numpy as np
import logging
import itertools

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

logging.basicConfig(level=logging.INFO)

# Load MNIST digits
data = sklearn.datasets.fetch_mldata('MNIST original')
X = data.data  # (n_examples, n_features)
n_examples, n_features = X.shape

################### Extract patches ###########################################

# Sample from data and reshape them into 2D images
sample_size = 1000
w = int(X.shape[1]**.5); digit_size = (w, w)
indices = random.sample(range(len(X)), sample_size)
digits = (X[ind].reshape(digit_size) for ind in indices)

# Create patches
patch_size = tuple(11 for _ in range(2))
patches = (patch
        for digit in digits
        for patch in extract_patches_2d(digit, patch_size, max_patches=10))

#canvas = util.tile(itertools.islice(patches, 100))
#plt.imshow(canvas)
#plt.show()
#assert False

kmeans = MiniBatchKMeans(n_clusters=100, verbose=True)

def display_components(V):
    plt.figure(figsize=(4.2, 4))

    patch_width = int(len(V[0])**.5)
    patches = [row.reshape((patch_width, patch_width)) for row in V]
    canvas = util.tile(patches)
    plt.imshow(canvas, interpolation='nearest', cmap=cm.gray)

batch_size = 1000
for curr_patches in util.chunks_of_size_n(patches, batch_size):
    data = np.vstack([patch.ravel() for patch in curr_patches])
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    kmeans.partial_fit(data)

display_components(kmeans.cluster_centers_)
plt.show()
