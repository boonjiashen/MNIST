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

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

logging.basicConfig(level=logging.INFO)

# Load MNIST digits
data = sklearn.datasets.fetch_mldata('MNIST original')
X = data.data  # (n_examples, n_features)
n_examples, n_features = X.shape

# Sample from digits
sample_size = 10000
sample = X[random.sample(range(n_examples), sample_size), :]
#sample = sample.reshape(-1, 28, 28)

patches = sample
np.random.shuffle(patches)
patch_size = (28, 28)

minibatch_kwargs = {
        'n_components': 100,
        'alpha': 1.,
        #'n_iter': 1,
        }
dico = MiniBatchDictionaryLearning(**minibatch_kwargs)
print('Learning the dictionary with')
for key, value in minibatch_kwargs.items(): print('\t', key, value)

def display_components(V):
    plt.figure(figsize=(4.2, 4))

    patches = [row.reshape(patch_size) for row in V]
    canvas = util.tile(patches)
    plt.imshow(canvas, interpolation='nearest', cmap=cm.gray)

batch_size = 1000  # Number of patches per batch
dico.n_iter = 1  # Training on each minibatch is just 1 iteration
n_batches = (len(patches) // batch_size)

logging.info('Training %i patches in batches of %i' %  \
        (len(patches), batch_size))
for bi in range(n_batches):
    print(bi, end=' ', flush=True)
    curr_patches = patches[bi * batch_size: (bi+1) * batch_size]
    V = dico.partial_fit(curr_patches, iter_offset=bi).components_

    display_components(V)

plt.show()
