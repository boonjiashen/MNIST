"""Average pooling on a matrix unrolled into a 1D vector

Transforms (n_samples, height * width) ->
(n_samples, height * width / 4) where each 2x2 cell is
reduced to the average value in the cell
"""

import numpy as np
import math

def get_transformation_matrix(X_shape):
    """`X_shape` is (height, width), the size of the matrix that X was unrolled
    from. height and width should be both divisible by 2.
    """

    H, W = X_shape
    assert H % 2 == 0 and W % 2 == 0

    if H == 2:
        submatrix = np.array([row
            for row in np.eye(W / 2, dtype=np.uint8)
            for _ in range(2)]).T
        trans_matrix = np.hstack([submatrix, submatrix])

        return trans_matrix
    
    nonzero_submat = get_transformation_matrix((2, W))

    # Create a list of list of submatrices, kind of like cells in MATLAB which
    # we'll then concatenate both horizontally and vertically to make a big
    # matrix
    submats = [[nonzero_submat if eye == 1 else np.zeros_like(nonzero_submat)
        for eye in row]
        for row in np.eye(H / 2)]

    trans_mat = np.vstack([np.hstack(row) for row in submats])

    return trans_mat


def transformation_input_output_pairs():
    """Generates a tuple of input output pairs. The outpool is the ave pooling
    result of the input. Used to check for correctness.
    """

    X = np.array([
            [4, 4],
            [4, 4]])
    Xprime = np.array([[4]])

    yield X, Xprime

    X = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2]])
    Xprime = np.array([[1, 2]])

    yield X, Xprime

    X = np.array([
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3]])
    Xprime = np.array([[1, 2, 3]])

    yield X, Xprime

    X = np.array([
            [1, 1],
            [1, 1],
            [3, 3],
            [3, 3],
            [6, 6],
            [6, 6]])
    Xprime = np.array([
        [1],
        [3], 
        [6]])

    yield X, Xprime

    X = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]])
    Xprime = np.array([
        [1, 2],
        [3, 4]])

    yield X, Xprime


def test_get_transformation_matrix():

    for X, expected_Xprime in transformation_input_output_pairs():

        H, W = X.shape  # (big) height and width before pooling

        # Unroll X into 1D vector and perform average pooling transformation
        M = get_transformation_matrix((H, W))  # matrix of 1s and 0s
        actual_Xprime = np.dot(M, X.ravel()) / 4  # divide by cell size

        # Roll X' back into a 2D matrix
        actual_Xprime = np.reshape(actual_Xprime, (H / 2, W / 2))

        np.testing.assert_almost_equal(actual_Xprime, expected_Xprime)
