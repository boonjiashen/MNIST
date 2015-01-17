"""Max pooling on a list of images

Transforms (n_samples, height, width, n_channels) ->
(n_samples, height/2, width/2, n_channels) where each 2x2 cell is
reduced to the maximum value in the cell
"""

import sklearn.base
import numpy as np
import math

def max_pool(im, step_size=2, chop_tail=True):
    """`im` is a (height, width, n_channels) matrix
    We reduce it to (height/2, width/2, n_channels) where each 2x2 cell is
    reduced to the maximum value in the cell

    `step_size` is an integer N where N >= 2 and (N, N) is the cell size.

    `chop_tail` if True, elements that fall outside of step_size x step_size
    cells are ignored.
    """

    if not chop_tail:

        # Pad image so that its size is a multiple of step_size
        h, w = im.shape[:2]
        desired_h = int(np.ceil(h / step_size)) * step_size
        desired_w = int(np.ceil(w / step_size)) * step_size
        im = np.pad(im, ((0, desired_h-h), (0, desired_w-w)), mode='minimum')

    h, w = np.array(im.shape[:2]) // step_size * step_size

    downsamples = [im[j:h:step_size, i:w:step_size]
            for j in range(step_size)
            for i in range(step_size)]

    return np.max(np.array(downsamples), axis=0)


class MaxPool(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, cell_width=2):
        """`cell_width` is the width of the square max-pool cell"""
        self.cell_width = cell_width

    def transform(self, X):
        return np.array([max_pool(x, step_size=self.cell_width) for x in X])


if __name__ == '__main__':
    """Example of max_pool()"""

    im = np.random.randint(0, 9, (7, 7))
    print('Target image is\n', im)
    for chop_tail in [True, False]:
        for step in [2, 3]:
            print('For step size', step, 'and chop_tail is', chop_tail, 'result is')
            print(max_pool(im, step_size=step, chop_tail=chop_tail))

