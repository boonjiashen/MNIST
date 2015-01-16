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
import sklearn.preprocessing
import sklearn.decomposition
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

    # Randomly sample a subset of the data
    sample_size = int(args.data_proportion * len(X))
    inds = random.sample(range(len(X)), sample_size)
    X, y = X[inds], y[inds]
    logging.info('Sampled %.1f%% of MNIST dataset', 100 * args.data_proportion)

    ############################# Define pipeline #############################    

    # Define whitener
    whiten_kwargs = {'whiten':True, 'copy':True}
    whitener = sklearn.decomposition.PCA(**whiten_kwargs)
    for key, value in whiten_kwargs.items():
        logging.info('Whitening kwarg, {0} = {1}'.format(key, value))

    # Define dictionary learner
    dic_kwargs = {'n_clusters': 100}
    dic = sklearn.cluster.KMeans(**dic_kwargs)
    for key, value in dic_kwargs.items():
        logging.info('Dictionary kwarg, {0} = {1}'.format(key, value))


    ############################# Generate patches from MNIST #################

    # Reshape them into 2D images
    w = int(X.shape[1]**.5); digit_size = (w, w)
    digits = X.reshape(len(X), digit_size[0], -1)

    # Create patches
    patch_size = (8, 8)
    patch_squares = np.vstack(
            extract_patches_2d(digit, patch_size, max_patches=10)
            for digit in digits)
    logging.info('Generated {0} patches of size {1}'.format(
        len(patch_squares), str(patch_size)))

    # Reshape each patch into one row vector
    patch_rows = patch_squares.reshape(len(patch_squares), -1)


    ######################### Pre-processing ##################################

    logging.info('Pre-processing')

    # Feature normalization
    patch_rows = patch_rows.astype(float)  # Cast as float (required by scale())
    patch_rows = sklearn.preprocessing.scale(patch_rows)

    # Whiten dataset
    patch_rows = whitener.fit_transform(patch_rows)

    logging.info('Done')


    ########################## Train clustering algorithm ##################### 

    # Train
    logging.info('Fitting %i patches', len(patch_rows))
    dic.fit(patch_rows)
    logging.info('done')


    ######################### Display atoms of dictionary #####################

    # Inverse whiten atoms of dictionary
    atom_rows = dic.cluster_centers_ 
    atom_rows = whitener.inverse_transform(atom_rows)  

    # Reshape to square
    atom_squares = atom_rows.reshape(len(atom_rows), patch_size[0], -1)

    plt.figure()
    for i, patch in enumerate(atom_squares):
        plt.subplot(10, 10, i + 1)
        plt.imshow(patch, cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Atoms of dictionary learnt from %i patches' %  \
            len(patch_rows))

    plt.show()
