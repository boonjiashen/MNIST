"""Encodes patches of a list of images.

We use this class to ensapsulate all the messy reshaping issues.

Transforms (n_samples, n_patches_per_sample, n_patch_features) ->
(n_samples, n_patches_per_sample, code_length_per_patch)
"""

import sklearn.base
import numpy as np

class Encoder(sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin):
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


