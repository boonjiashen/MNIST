"""Max pooling on a list of images

Transforms (n_samples, height, width, n_channels) ->
(n_samples, height/2, width/2, n_channels) where each 2x2 cell is
reduced to the maximum value in the cell
"""

import sklearn.base
import numpy as np

class MaxPool(sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin):
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


