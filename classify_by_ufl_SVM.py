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

import sklearn.datasets
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.pipeline
import sklearn.feature_extraction.image as image
import sklearn.base


class MaxPool(sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin):
    """Max pooling on a list of images

    Transforms (n_samples, height, width, n_channels) ->
    (n_samples, height/2, width/2, n_channels) where each 2x2 cell is
    reduced to the maximum value in the cell
    """

    def _max_pool(self, im):
        """`im` is a (height, width, n_channels) matrix

        We reduce it to (height/2, width/2, n_channels) where each 2x2 cell is
        reduced to the maximum value in the cell
        """
        h, w = im.shape[:2]
        assert h%2==0 and w%2==0, "Illegal image dimensions"

        tl = im[::2, ::2]
        tr = im[::2, 1::2]
        bl = im[1::2, ::2]
        br = im[1::2, 1::2]

        pool = np.max(np.array([tl, tr, bl, br]), axis=0)

        return pool

    def transform(self, X):
        return np.array([self._max_pool(x) for x in X])


class Encoder(sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin):
    """Encodes patches of a list of images.
    
    We use this class to ensapsulate all the messy reshaping issues.

    Transforms (n_samples, n_patches_per_sample, n_patch_features) ->
    (n_samples, n_patches_per_sample, code_length_per_patch)
    """

    def __init__(self, coder):
        """`coder` has a transform method that encodes a
        (n_signals, n_features) -> (n_components, n_features)
        """
        self.coder = coder

    def transform(self, X):
        # `patch_rows` is a (n_patches_per_sample, n_patch_features) matrix
        digits_as_patch_rows = X
        return np.array([
            self.coder.transform(patch_rows)
            for patch_rows in digits_as_patch_rows])


class PatchExtractor(sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin):
    """Extract patches for images while maintaining a dimension for images.

    We use this class to ensapsulate all the messy reshaping issues.

    Transforms (n_samples, image_height, image_width) to
    (n_samples, n_patches_per_sample, n_patch_features)
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.patchifier = image.PatchExtractor(patch_size)

    def transform(self, digit_squares):
        # (n_samples, image_height, image_width) ->
        # (n_patches, patch_height, patch_width)
        patch_squares = self.patchifier.transform(digit_squares)

        logging.debug(patch_squares.shape)

        # (n_patches, patch_height, patch_width) ->
        # (n_samples, n_patches_per_image, patch_height, patch_width)
        n_samples = len(digit_squares)
        n_patches_per_sample = len(patch_squares) / n_samples
        digits_as_patch_rows = patch_squares.reshape(
                n_samples, n_patches_per_sample, -1)

        logging.debug(digits_as_patch_rows.shape)

        return digits_as_patch_rows

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
    patchifier = PatchExtractor(patch_size)
    coder = Encoder(sklearn.decomposition.SparseCoder(dictionary,
        transform_algorithm='threshold'))
    maxpool = MaxPool()

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
