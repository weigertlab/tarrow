import pathlib
import logging
import sys
from time import time as now
import random
from itertools import product, chain
from typing import Sequence, Union

import numpy as np
import torch


logger = logging.getLogger(__name__)


def normalize(
    x, pmin=1, pmax=99.8, clip=False, eps=1e-10, axis=None, subsample: int = 1
):
    x = np.array(x, dtype=np.float32, copy=False)

    # standardize axis, e.g. (-2,-1) -> (1,2) for 3d data
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        axis = tuple(np.arange(x.ndim)[axis])

    # if subsample > 1, use a fraction for percentile calculation (faster)
    subslice = tuple(
        slice(0, None, subsample) if i in axis else slice(None)
        for i in tuple(range(x.ndim))
    )

    mi, ma = np.percentile(x[subslice], (pmin, pmax), axis=axis, keepdims=True)
    logger.debug(f"Min intensity (at p={pmin/100}) = {mi}")
    logger.debug(f"Max intensity (at p={pmax/100}) = {ma}")

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x.astype(np.float32, copy=False)


def crop(x, divby):
    assert len(x) == len(divby)
    return x[tuple(slice(0, (s // d) * d) for s, d in zip(x.shape, divby))]


def seed(s=None):
    """Seed random number generators.

    Defaults to unix timestamp of function call.

    Args:

        s (``int``):

            Manual seed.
    """

    if s is None:
        s = int(now())

    random.seed(s)
    logger.debug(f"Seed `random` rng with {s}.")
    np.random.seed(s)
    logger.debug(f"Seed `numpy` rng with {s}.")
    if "torch" in sys.modules:
        torch.manual_seed(s)
        logger.debug(f"Seed `torch` rng with {s}.")

    return s


def glob_multiple(
    fname: Union[pathlib.Path, str], patterns: Sequence[str] = ("*.tif", "*.png")
):
    fname = pathlib.Path(fname)
    return tuple(chain(*tuple(fname.glob(p) for p in patterns)))


def crop(x, shape, spatial_dims=2):
    """Center-crop x to match spatial dimensions given by shape."""

    x_target_size = x.size()[:-spatial_dims] + shape
    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))
    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def tile_iterator(
    im, blocksize=(64, 64), padsize=(64, 64), mode="constant", verbose=False
):
    """

    iterates over padded tiles of an ND image
    while keeping track of the slice positions

    Example:
    --------
    im = np.zeros((200,200))
    res = np.empty_like(im)

    for padded_tile, s_src, s_dest in tile_iterator(im,
                              blocksize=(128, 128),
                              padsize = (64,64),
                              mode = "wrap"):

        #do something with the tile, e.g. a convolution
        res_padded = np.mean(padded_tile)*np.ones_like(padded_tile)

        # reassemble the result at the correct position
        res[s_src] = res_padded[s_dest]



    Parameters
    ----------
    im: ndarray
        the input data (arbitrary dimension)
    blocksize:
        the dimension of the blocks to split into
        e.g. (nz, ny, nx) for a 3d image
    padsize:
        the size of left and right pad for each dimension
    mode:
        padding mode, like numpy.pad
        e.g. "wrap", "constant"...

    Returns
    -------
        tile, slice_src, slice_dest

        tile[slice_dest] is the tile in im[slice_src]

    """

    if not (im.ndim == len(blocksize) == len(padsize)):
        raise ValueError(
            "im.ndim (%s) != len(blocksize) (%s) != len(padsize) (%s)"
            % (im.ndim, len(blocksize), len(padsize))
        )

    subgrids = tuple([int(np.ceil(1.0 * n / b)) for n, b in zip(im.shape, blocksize)])

    # if the image dimension are not divible by the blocksize, pad it accordingly
    pad_mismatch = tuple(
        [(s * b - n) for n, s, b in zip(im.shape, subgrids, blocksize)]
    )

    if verbose:
        print("tile padding... ")

    im_pad = np.pad(
        im, [(p, p + pm) for pm, p in zip(pad_mismatch, padsize)], mode=mode
    )

    # iterates over cartesian product of subgrids
    for i, index in enumerate(product(*[range(sg) for sg in subgrids])):
        # dest[s_output] is where we will write to
        s_input = tuple([slice(i * b, (i + 1) * b) for i, b in zip(index, blocksize)])

        s_output = tuple(
            [
                slice(p, b + p - pm * (i == s - 1))
                for b, pm, p, i, s in zip(
                    blocksize, pad_mismatch, padsize, index, subgrids
                )
            ]
        )

        s_padinput = tuple(
            [
                slice(i * b, (i + 1) * b + 2 * p)
                for i, b, p in zip(index, blocksize, padsize)
            ]
        )
        padded_block = im_pad[s_padinput]

        yield padded_block, s_input, s_output


def uniform_filter(x, k):
    assert k % 2 == 1
    nc = x.shape[1]
    wx = torch.ones(nc, 1, k, 1).to(x.device)
    wy = torch.ones(nc, 1, 1, k).to(x.device)
    x = torch.nn.functional.conv2d(x, wx, groups=nc, padding=(k // 2, 0))
    x = torch.nn.functional.conv2d(x, wy, groups=nc, padding=(0, k // 2))
    x = x / k**2
    return x
