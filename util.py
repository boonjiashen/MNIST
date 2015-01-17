"""Useful functions
"""

import math
import numpy as np
import logging


def yield_windows(image, window_size, step_size, yield_bb=False):
    """Yield windows of an image in regular intervals in row-major order.

    `image` - a 2D image

    `window_size` - required (height, width) of window

    `step_size` - (vertical_step, horizontal_step) 2-ple

    `yield_bb' - yields the bounding box of the window if True, i.e., yields a
    (window, (xTL, yTL, xBR, yBR)) tuple, where TL and BR are top-left and
    bottom-right of the window.
    """

    im_height, im_width = image.shape[:2]
    win_height, win_width = window_size
    y_step, x_step = step_size

    # y coord of TL of bottom-most window
    max_y_TL = (im_height - win_height) // y_step * y_step

    # x coord of TL of left-most window
    max_x_TL = (im_width - win_width) // x_step * x_step

    for y_TL in range(0, max_y_TL + 1, y_step):
        for x_TL in range(0, max_x_TL + 1, x_step):
            window = image[
                    y_TL:y_TL + win_height,
                    x_TL:x_TL + win_width]

            # Yield both the window and its coordinates
            if yield_bb:
                bb = (x_TL, y_TL, x_TL + win_width - 1, y_TL + win_height - 1)
                yield window, bb

            # Yield window only
            else:
                yield window


def chunks_of_size_n(iterator, n):
    "Generators generators each of size n"

    def chunk():
        yield next_item
        for _ in range(n - 1):
            yield next(iterator)

    if not hasattr(iterator, '__iter__'):
        raise IOError('argument is not an iterator')
    else:
        iterator = iter(iterator)
    while True:
        try:
            next_item = next(iterator)
            yield chunk()
        except StopIteration:
            return


def tile(tiles, desired_aspect=1., border_width=0):
    """Return a canvas from tiling 2D images of the same size

    Tries to return an image as square as possible.

    `tiles` generator or iterator of 2D images of the same size

    `desired_aspect` = width/height, desired aspect ratio of canvas, e.g. 16/9
    when a screen is 16:9
    """

    def unaspectness(tile_size, tiling_factor, desired_aspect=1.):
        """A metric of how close a 2D image is to an aspect ratio when it is
        tiled in both dimensions.

        The smaller the metric, the more square the tiling pattern is.

        `tile_size` = (height, width) size of tile

        `tiling_factor` = (vertical_factor, horizontal_factor) no. of times the tile
        is repeated in each direction

        `desired_aspect` = width/height, desired aspect ratio, e.g. 16/9 when a
        screen is 16:9
        """

        # Height and width of final tiled pattern
        h, w = [sz * factor for sz, factor in zip(tile_size, tiling_factor)]

        # We square the log of the ratios so that unsquaredness of 1/x or x is the
        # same
        unaspectness = math.log(w/h/desired_aspect)**2

        return unaspectness 

    tiles = list(tiles)

    # Make sure that all tiles share the same size
    for tile in tiles:
        assert tile.shape == tiles[0].shape

    # Get optimal tiling factor
    n_tiles = len(tiles)
    tile_size = tiles[0].shape
    tiling_factor = min(
            ((math.ceil(n_tiles / i), i) for i in range(1, n_tiles + 1)),
            key=lambda x: unaspectness(tile_size, x, desired_aspect)
            )

    # Add blank tiles to fill up canvas
    blank_tile = np.zeros_like(tiles[0])
    tiles.extend([blank_tile for i in range(np.prod(tiling_factor) - n_tiles)])

    # Add borders
    if border_width != 0:
        tiles = [np.pad(tile, ((0, border_width), (0, border_width)),
                mode='constant')
                for tile in tiles]

    # Tile tiles
    rows = [np.hstack(tiles[i:i+tiling_factor[1]])
        for i in range(0, len(tiles), tiling_factor[1])]
    canvas = np.vstack(rows)

    return canvas


def demo_tile():
    """Demo tile() with stdin

    Instructions: press ESC to exit
    """

    import cv2
    import itertools
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    args = parser.parse_args()

    # Grab all frames
    full_frames = grab_frame(args.video_filename)

    # Resize all frames to fit better into canvas
    scale = .3  # Resize scale
    mini_frames = (cv2.resize(x, None, fx=scale, fy=scale)
            for x in full_frames)

    # Select every N frames
    step, n_frames = 15, 40
    frames = itertools.islice(mini_frames, 0, n_frames * step, step)

    # Tile selected frames into a canvas
    canvas = tile(frames, desired_aspect=16/9)

    cv2.imshow('1', canvas)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_chunks_of_size_n():
    """Example of chunks_of_size_n()"""

    import itertools
    iterator = range(10)
    for items in chunks_of_size_n(iterator, 3):
        print(list(items))


if __name__ == "__main__":
    example = example_chunks_of_size_n
    print(example.__name__)
    example()
