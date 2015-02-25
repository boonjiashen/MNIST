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
import CoatesScaler
import ZCA
import pickle

import sklearn.datasets
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.pipeline
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def print_steps(steps, printer=logging.info):
    """Pretty-print steps of a pipeline

    `steps` is a list of (class_object, dict) tuples

    `printer` is the function used to print
    """

    # Print steps and respective kwargs in pipeline
    for si, (class_object, kwargs) in enumerate(steps, 1):
        if not kwargs:
            printer('{}) {}'.format(si, class_object.__name__))
            continue

        # Width of kwarg keyword, to make sure they right-justify
        width = max(map(len, kwargs.keys()))

        for ki, (key, value) in enumerate(kwargs.items()):
            fmt = '{} {:>%i} = {}' % width
            info = fmt.format(class_object.__name__, key, value)

            # Add step index (or blank space to maintain column format)
            if ki == 0:
                info = '{}) '.format(si) + info
            else:
                info = '   ' + info
            printer(info)


if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_proportion", nargs='?', type=float, default=1.,
            help="Proportion of full MNIST dataset to be used")
    parser.add_argument("--output_pickle", type=str,
            help="Filename to pickle (n_components, n_features)-sized dictionary into")
    parser.add_argument("--log", type=str, default='INFO',
            help="Logging setting (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    # Setting logging parameters
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')

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

    std_scaler = (sklearn.preprocessing.StandardScaler, {})
    coates_scaler = (CoatesScaler.CoatesScaler, {})
    pca = (sklearn.decomposition.PCA,
            {'whiten':True, 'copy':True}
            )
    zca = (ZCA.ZCA, {'regularization': .1})
    mbkmeans = (sklearn.cluster.MiniBatchKMeans,
            {
                'n_clusters': 100,
                'batch_size': 3000,
            })
    kmeans = (sklearn.cluster.KMeans,
            {
                'n_clusters': 100,
                #'n_jobs': -1,
                'n_init': 1,
                'max_iter': 10,
            })

    # Define pipeline
    steps = [coates_scaler, zca, kmeans]
    pipeline = sklearn.pipeline.make_pipeline(
            *[fun(**kwargs) for fun, kwargs in steps])

    # Define pointers to certain steps for future processing
    whitener = pipeline.steps[1][1]  # second step
    dic = pipeline.steps[-1][1]  # last step

    print_steps(steps)

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


    ######################### Train pipeline ##################################

    logging.info('Training model...')
    pipeline.fit(patch_rows.astype(float))
    logging.info('done.')


    ######################### Display atoms of dictionary #####################

    logging.info('Displaying atoms of dictionary')

    # Inverse whiten atoms of dictionary
    atom_rows = dic.cluster_centers_ 
    if hasattr(whitener, 'inverse_transform'):
        atom_rows = whitener.inverse_transform(atom_rows)  

    # Reshape to square
    atom_squares = atom_rows.reshape(len(atom_rows), patch_size[0], -1)

    plt.figure()
    for i, patch in enumerate(atom_squares):
        plt.subplot(10, 10, i + 1)
        plt.imshow(patch, cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Atoms of dictionary learnt from %i patches by %s' %  \
            (len(patch_rows), dic.__class__.__name__))

    plt.show()


    ######################### Pickle dictionary if required ###################
    
    atom_rows = dic.cluster_centers_ 
    if args.output_pickle is not None:
        with open(args.output_pickle, 'wb') as fid:
            pickle.dump(atom_rows, fid)

        logging.info('Pickled %s-sized dictionary into %s',
                str(atom_rows.shape), args.output_pickle)
