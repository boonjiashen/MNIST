"""Classify by encoding an image via a dictionary and running SVM on the
resultant sparse code.
"""

# Suppress deprecation warnings that come from sklearn's omp.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import time
import numpy as np
import logging
import itertools
import argparse
import pickle

import CoatesScaler
import ZCA
import PatchExtractor
import MaxPool
import Encoder
import util

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
    parser.add_argument("--input_pickle", type=str,
            help="File to unpickle dictionary from (default: populate dictionary with random patches")
    args = parser.parse_args()

    # Load MNIST digits
    data = sklearn.datasets.fetch_mldata('MNIST original')
    X = data.data  # (n_examples, n_features)
    y = data.target  # (n_examples)

    X = X.astype(float)

    # Get size of a digit image
    w = int(X.shape[1]**.5); digit_size = (w, w)

    ############################# Generate dictionary from random patches #####

    if args.input_pickle is None:

        # Reshape sample of digits into 2D images
        n_atoms = 100
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

    else:

        with open(args.input_pickle, 'rb') as fid:
            dictionary = pickle.load(fid)

        n_atoms = dictionary.shape[0]
        w = int(dictionary.shape[1]**.5); patch_size = (w, w)


    ######################### Extract patches from one image ##################

    # Define pipeline
    coates_scaler = CoatesScaler.CoatesScaler()
    zca = ZCA.ZCA(regularization=.1)
    patchifier = PatchExtractor.PatchExtractor(patch_size)
    patch_coder = sklearn.decomposition.SparseCoder(dictionary)
    digit_coder = Encoder.Encoder(patch_coder)
    maxpool = MaxPool.MaxPool()

    # Generate inputs
    n_samples = int(len(X) * args.data_proportion)
    X_rows = X[random.sample(range(len(X)), n_samples), :]

    X_squares = X_rows.reshape(len(X_rows), digit_size[0], -1)

    # Crop digits so that n_patches per dimension is a multiple of 2
    X_squares = X_squares[:, :27, :27]

    logging.info('Transforming %i digits from MNIST', len(X_rows))

    patch_squares = np.vstack(
            image.extract_patches_2d(digit, patch_size)
            for digit in X_squares)
    logging.info('Generated {0} patches of size {1}'.format(
        len(patch_squares), str(patch_size)))

    # Reshape each patch into one row vector (n_patches, numel_per_patch)
    patch_rows = patch_squares.reshape(len(patch_squares), -1)

    logging.info('Feature normalization and whitening...')

    # Feature normalization, whitening
    tmp = patch_rows
    tmp = coates_scaler.fit(tmp).transform(tmp)
    tmp = zca.fit(tmp).transform(tmp)
    patch_rows = tmp

    logging.info('Encoding patches...')

    # Encode patches into codes (n_codes, n_atoms)-shape
    codes = patch_coder.transform(patch_rows)

    logging.info('Max pooling...')

    # Reshape codes as squares for max pooling
    n_samples = len(X_rows)
    patch_grid_width = (len(patch_rows) // n_samples)**.5
    X_code_squares = codes.reshape(
            n_samples, patch_grid_width, patch_grid_width, -1)

    # Perform max-pooling
    X_code_pool = maxpool.transform(X_code_squares)

    X_code_pool_rows = X_code_pool.reshape(len(X_code_pool), -1)

    if True:
        # Display each feature of a single digit
        n_displays = 5
        inds = random.sample(range(len(X_code_squares)), n_displays)
        code_squares = X_code_squares[inds]
        for i, code_square in enumerate(code_squares, 1):
            logging.info('Displaying code example %i', i)
            plt.figure()
            for channel_n in range(code_square.shape[-1]):
                plt.subplot(10, 10, channel_n + 1)
                channel = code_square[:, :, channel_n]

                plt.imshow(channel, cmap=plt.cm.gray, interpolation='nearest')
                plt.xticks(())
                plt.yticks(())

        plt.show()
