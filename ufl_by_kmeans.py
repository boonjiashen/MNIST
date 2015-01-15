"""Learn a dictionary from MNIST digits
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import util
import time
import numpy as np
import logging
import itertools
import argparse

import sklearn.datasets
import sklearn.cluster
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

if __name__ == "__main__":

    # Set logging parameters
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s')

    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_proportion", nargs='?', type=float, default=1.,
            help="Proportion of full MNIST dataset to be used")
    args = parser.parse_args()

    # Load MNIST digits
    data = sklearn.datasets.fetch_mldata('MNIST original')
    X = data.data  # (n_examples, n_features)
    y = data.target  # (n_examples)


    ############################# Generate patches from MNIST #################

    # Randomly sample a subset of the data
    sample_size = int(args.data_proportion * len(X))
    inds = random.sample(range(len(X)), sample_size)
    X, y = X[inds], y[inds]
    logging.info('Sampled %.1f%% of MNIST dataset', 100 * args.data_proportion)

    # Reshape them into 2D images
    w = int(X.shape[1]**.5); digit_size = (w, w)
    indices = random.sample(range(len(X)), sample_size)
    #indices = range(len(X))
    digits = (X[ind].reshape(digit_size) for ind in indices)

    # Create patches
    patch_size = tuple(15 for _ in range(2))
    patches = (patch
            for digit in digits
            for patch in extract_patches_2d(digit, patch_size, max_patches=10))

    dic = sklearn.cluster.MiniBatchKMeans(n_clusters=100, verbose=True)

    def display_components(V):
        plt.figure()

        patch_width = int(len(V[0])**.5)
        patches = [row.reshape((patch_width, patch_width)) for row in V]
        canvas = util.tile(patches)
        #plt.imshow(canvas, interpolation='nearest', cmap=cm.gray)
        plt.imshow(canvas, interpolation='nearest')

    batch_size = 1000
    for curr_patches in util.chunks_of_size_n(patches, batch_size):
        data = np.vstack([patch.ravel() for patch in curr_patches])
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        dic.partial_fit(data)

    display_components(dic.cluster_centers_)
    plt.show()
