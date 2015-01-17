"""Classify by encoding an image via a dictionary and running SVM on the
resultant sparse code.
"""

# Suppress deprecation warnings that come from sklearn's omp.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import util
import time
import numpy as np
import logging
import itertools
import argparse
import CoatesScaler
import ZCA
import PatchExtractor
import MaxPool
import Encoder

import sklearn.datasets
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.pipeline
import sklearn.feature_extraction.image as image
import sklearn.base


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

    X = X.astype(float)


    ############################# Generate dictionary from random patches #####

    # Reshape sample of digits into 2D images
    n_atoms = 100
    w = int(X.shape[1]**.5); digit_size = (w, w)
    digit_squares = (X[i].reshape(digit_size)
            for i in random.sample(range(len(X)), n_atoms))

    # Grab one random patch from each digit
    patch_size = (8, 8)
    patch_squares = np.vstack(
            image.extract_patches_2d(digit, patch_size, max_patches=1)
            for digit in digit_squares
            )

    # Flatten patch squares to (n_components, n_features)
    dictionary = patch_squares.reshape(len(patch_squares), -1)


    ######################### Extract patches from one image ##################

    # Define pipeline
    patchifier = PatchExtractor.PatchExtractor(patch_size)
    coder = Encoder.Encoder(sklearn.decomposition.SparseCoder(dictionary,
        transform_algorithm='threshold'))
    maxpool = MaxPool.MaxPool()

    # Generate inputs
    n_samples = int(len(X) * args.data_proportion)
    X_rows = X[:n_samples, :]
    X_squares = X_rows.reshape(len(X_rows), digit_size[0], -1)

    # Crop digits so that n_patches per dimension is a multiple of 2
    X_squares = X_squares[:, :27, :27]

    logging.info('Transforming %i samples', len(X_rows))
    logging.info('Creating patches...')

    # Reshape patch squares to rows to allow sparse coding
    X_patch_rows = patchifier.transform(X_squares)

    logging.info('Encoding patches...')

    # Encode each patch
    X_code_rows = coder.transform(X_patch_rows)

    # Reshape patch codes as squares for max pooling
    n_samples = X_patch_rows.shape[0]
    w = int(X_patch_rows.shape[1]**.5); patch_grid_size = (w, w)
    X_code_squares = X_code_rows.reshape(
            n_samples, patch_grid_size[0], patch_grid_size[0], -1)

    logging.info('Max pooling...')

    X_code_pool = maxpool.transform(X_code_squares)

    X_code_pool_rows = X_code_pool.reshape(len(X_code_pool), -1)

    logging.info(X_patch_rows.shape)
    logging.info(X_code_rows.shape)
    logging.info(X_code_squares.shape)
    logging.info(X_code_pool.shape)
    logging.info(X_code_pool_rows.shape)

    if False:
        # Display each feature of a single digit
        code_square = (X_code_squares[0])
        plt.figure()
        for channel_n in range(code_square.shape[-1]):
            plt.subplot(10, 10, channel_n + 1)
            channel = code_square[:, :, channel_n]

            plt.imshow(channel, cmap=plt.cm.gray, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

        plt.show()
