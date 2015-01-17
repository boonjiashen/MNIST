"""Extract patches for images while maintaining a dimension for images.

We use this class to ensapsulate all the messy reshaping issues.

Transforms (n_samples, image_height, image_width) to
(n_samples, n_patches_per_sample, n_patch_features)
"""
import logging

import sklearn.base
import sklearn.feature_extraction.image as image

class PatchExtractor(sklearn.base.BaseEstimator,
        sklearn.base.TransformerMixin):

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

