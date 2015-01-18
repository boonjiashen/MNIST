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
from ufl_by_kmeans import print_steps

import sklearn.datasets
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.pipeline
import sklearn.feature_extraction.image as image
import sklearn.base
import sklearn.svm
import sklearn.cross_validation

def to_transformer_class(fun):
    """Turns a function into an sklearn transformer class with the function
    as the transform method of the class

    `fun` takes a single argument X and returns the transformation of X
    """
    class Transform(sklearn.base.BaseEstimator,
            sklearn.base.TransformerMixin):
        def fit(self, X): return self
        def transform(self, X): return fun(X)
    Transform.__name__ = fun.__name__[0].upper() + fun.__name__[1:]
    return Transform


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
    parser.add_argument("--n_display", type=int, default=0,
            help="Number of output variables to be displayed.")
    args = parser.parse_args()

    # Load MNIST digits
    data = sklearn.datasets.fetch_mldata('MNIST original')
    all_digits = data.data  # (n_examples, n_features)
    all_targets = data.target  # (n_examples)

    all_digits = all_digits.astype(float)

    # Get size of a digit image
    w = int(all_digits.shape[1]**.5); digit_size = (w, w)

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

    # Get patch size (as many elements as atom length)
    w = int(dictionary.shape[1]**.5); patch_size = (w, w)


    ######################### Define pipeline #################################

    patch_step_size = 2
    Patchifier = to_transformer_class(lambda X:
                np.array([patch
                for digit in X
                for patch in util.yield_windows(digit, patch_size,
                    (patch_step_size, patch_step_size))])
                )

    # Find out patch grid width to figure out how to reshape codes into squares
    digit = all_digits[0].reshape(1, digit_size[0], -1)
    patches = Patchifier().transform(digit)
    patch_grid_width = int(len(patches)**.5)
    logging.info('Patchifier generates %i patches per digit', len(patches))

    def squarify_codes(X):
        """Reshapes (n_codes, n_atoms)-shape ->
        (n_samples, grid_width, grid_width, n_atoms)
        """
        n_codes, n_atoms = X.shape
        shape = (-1, patch_grid_width, patch_grid_width, n_atoms)
        logging.info('Squarify codes with shape %s', str(shape))
        return X.reshape(*shape)

    def flatten_except_axis0(X):
        """Reshapes (dim0, ...) -> (dim0, dim1)
        """
        return X.reshape(len(X), -1)

    # Define ad-hoc transformer classes
    CodeSquarer = to_transformer_class(squarify_codes)
    FlattenExceptAxis0 = to_transformer_class(flatten_except_axis0)
    RLU = to_transformer_class(lambda X:
            np.maximum(0, np.dot(X, dictionary.T)))
    RLU.__name__ = 'RLU'

    # Define possible steps to pipeline
    coates_scaler = (CoatesScaler.CoatesScaler, {})
    zca = (ZCA.ZCA, {'regularization':.1})
    patch_coder = (sklearn.decomposition.SparseCoder,
            {
                'dictionary':dictionary,
                'transform_algorithm':'threshold',
                'transform_alpha':0
            })
    flattener = (FlattenExceptAxis0, {})
    code_squarer = (CodeSquarer, {})
    maxpool = (MaxPool.MaxPool, {})
    rlu = (RLU, {})

    # Define pipeline
    steps = [coates_scaler, zca, rlu,
            code_squarer,
            maxpool,
            flattener]
    pipeline = sklearn.pipeline.make_pipeline(
            *[fun(**kwargs) for fun, kwargs in steps])

    print_steps(steps)



    ######################### Sample subset of MNIST ##########################

    n_samples = int(len(all_digits) * args.data_proportion)
    inds = random.sample(range(len(all_digits)), n_samples)
    X_rows, y = all_digits[inds], all_targets[inds]


    ######################### Extract patches #################################

    X_squares = X_rows.reshape(len(X_rows), digit_size[0], -1)

    logging.info('Transforming %i digits from MNIST', len(X_rows))

    patch_squares = Patchifier().transform(X_squares)

    logging.info('Generated {} patches of size {}, {} patches per digit'.format(
        len(patch_squares),
        str(patch_size),
        len(patch_squares) // len(X_squares)))

    # Reshape each patch into one row vector (n_patches, numel_per_patch)
    patch_rows = patch_squares.reshape(len(patch_squares), -1)


    ######################### Transformation ##################################

    logging.info('Pre-processing patches...')
    preprocessor = sklearn.pipeline.Pipeline(pipeline.steps[:2])
    whitened_patch_rows = preprocessor.fit_transform(patch_rows)

    # Encode patches into codes (n_codes, n_atoms)-shape
    logging.info('Encoding patches...')
    processor = sklearn.pipeline.Pipeline(pipeline.steps[2:])
    X_code_pool_rows = processor.transform(whitened_patch_rows)


    ######################### Train and test classifier #######################

    clf = sklearn.svm.LinearSVC()

    # Get F1 score by KFolds cross validation
    n_folds = 5
    f1s = sklearn.cross_validation.cross_val_score(clf, X_rows, y,
            scoring='f1', cv=n_folds)
    f1s *= 100

    print('Over {} folds, f1 = {:.1f}% +/- {:.1f}%'.format(
        n_folds, np.mean(f1s), np.std(f1s)))


    ######################### Display transformed digits ######################

    # Display each feature of several digits
    n_displays = args.n_display
    examples = random.sample(list(X_code_pool_rows), n_displays)
    for i, code_pool_row in enumerate(examples, 1):
        logging.info('Displaying code example %i', i)
        code_pool_square = code_pool_row.reshape(10, 10, -1)
        plt.figure()
        for channel_n in range(code_pool_square.shape[-1]):
            channel = code_pool_square[:, :, channel_n]

            plt.subplot(10, 10, channel_n + 1)
            plt.imshow(channel, cmap=plt.cm.gray, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

    plt.show()
