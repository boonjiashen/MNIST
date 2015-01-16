"""Useful functions
"""

import cv2
import math
import numpy as np

def chunks_of_size_n(iterator, n):
    "Split a generator into lists each of size n"

    def chunk():
        for i in range(n):
            yield next(iterator)

    while True:
        curr_chunk = list(chunk())
        if curr_chunk:
            yield curr_chunk
        else:
            raise StopIteration


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


if __name__ == "__main__":
    demo = demo_tile
    print(demo.__doc__)
    demo()
