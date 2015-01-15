"""Load MNIST and display random digits
"""

import matplotlib.pyplot as plt
import random
import util
import logging
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.datasets
import sklearn.metrics
import numpy as np
import argparse

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

    # Split dataset into training and test sets
    train_proportion = .6
    Xtrain, Xtest, ytrain, ytest = sklearn.cross_validation.train_test_split(
            X, y, train_size=train_proportion)
    logging.info('Split data set into %i training and %i test examples',
            len(Xtrain), len(Xtest))

    # Training
    logging.info('Training classifier')
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xtrain, ytrain)
    logging.info('done')

    # Testing
    logging.info('Testing classifier')
    predictions = clf.predict(Xtest)
    logging.info('done')

    # Print performance metrics
    metrics = 'f1_score', 'accuracy_score', 'precision_score', 'recall_score'
    for metric_name in metrics:
        score = getattr(sklearn.metrics, metric_name)(ytest, predictions)

        # Prettify metric name formatting
        metric_fmt = (('%' + str(max(map(len, metrics))) + 's') % metric_name)

        print(metric_fmt, '=', score)
