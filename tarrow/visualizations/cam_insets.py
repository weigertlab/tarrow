import logging
import argparse
from pathlib import Path

from scipy.ndimage import zoom

from skimage.io import imread
from skimage.feature import corner_peaks
from skimage.transform import resize
from skimage.color import rgb2gray

from tqdm import tqdm

import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

# from ..utils import normalize
from tarrow.utils import normalize


logger = logging.getLogger(__name__)


def cached_imread(f, norm=True, togray=False):
    x = imread(f)
    if togray:
        x = rgb2gray(x)
    if norm:
        x = normalize(x)
    return x


def _hvstack(x, ys, order=1):
    """horizontal stack x and vertically stacked ys"""

    if len(ys) > 1:
        x = _hvstack(x, ys[:-1], order=order)
        return _hvstack(x, ys[-1:])
    else:
        ys = ys[0]
    assert x.ndim in (2, 3)
    h, w = x.shape[:2]
    y = np.concatenate(ys, axis=0)
    new_shape = (h, int(y.shape[1] * h / y.shape[0]))

    if x.ndim == 3:
        new_shape = new_shape + (x.shape[-1],)
    y = resize(y, new_shape, order=order)
    z = np.concatenate((x, y), axis=1)
    return z


def _vhstack(x, ys, order=1):
    """vertical stack x and horizontally stacked ys"""

    if len(ys) > 1:
        x = _vhstack(x, ys[:-1], order=order)
        return _vhstack(x, ys[-1:])
    else:
        ys = ys[0]
    assert x.ndim in (2, 3)
    h, w = x.shape[:2]
    y = np.concatenate(ys, axis=1)
    new_shape = (h, int(y.shape[1] * h / y.shape[0]))
    new_shape = (int(y.shape[0] * w / y.shape[1]), w)

    if x.ndim == 3:
        new_shape = new_shape + (x.shape[-1],)
    y = resize(y, new_shape, order=order)
    z = np.concatenate((x, y), axis=0)
    return z


def _blend(x, y, alpha=0.5):
    # assert x.ndim==y.ndim==3
    return (1 - alpha) * x + alpha * y


def _fill_border(x, w, val=0):
    if x.ndim == 3:
        y = np.ones_like(x)
        y[..., :3] *= val
    else:
        y = val * np.ones_like(x)

    y[w:-w, w:-w] = x[w:-w, w:-w]
    return y


def get_peaks(y, scale=0.1, min_dist=10, th=None):
    y = zoom(y, scale, order=1)
    if th is None:
        th = np.percentile(y, 90)

    inds = corner_peaks(
        y, threshold_abs=th, min_distance=int(scale * min_dist), indices=True
    )
    inds = inds[np.argsort(y[tuple(inds.T)])[::-1]]
    inds = (1.0 / scale * inds).astype(int)
    return inds


def cam_insets(
    xs,
    cam,
    n_insets=5,
    w_inset=100,
    horiz=False,
    th=None,
    fig=None,
    ax=None,
    im=None,
    main_frame=0,
):
    if cam.ndim == 3:
        cam = cam[main_frame]
    if th is None:
        th = np.percentile(cam[::4, ::4], 90)
        logger.debug(f"{th=}")

    cmap_img = plt.cm.gray
    cmap_cam = plt.cm.magma

    xcs = tuple(cmap_img(x) for x in xs)
    camc = cmap_cam(normalize(cam))
    bcs = _blend(xcs[main_frame], camc, 0.5)

    # dont get inds from border
    inds = get_peaks(
        # +2 to fix off by one errors induced by scaling. This ensures that slices are contained in full image.
        _fill_border(cam, (w_inset + 2) // 2, 0),
        scale=0.3,
        min_dist=w_inset / 2,
        th=th,
    )[:n_insets]

    ss = tuple(
        tuple(slice(i - w_inset // 2, i + w_inset // 2) for i in ii) for ii in inds
    )
    scs = tuple(
        tuple(_fill_border(normalize(_x[s], clip=True), 2, 0.3) for s in ss)
        for _x in xcs
    )

    if len(inds) < n_insets:
        _empty = np.ones((w_inset, w_inset, 4))
        _empty[..., :3] = 0
        scs = tuple(s + (_empty,) * (n_insets - len(inds)) for s in scs)

    if horiz:
        final = _hvstack(bcs, scs)
    else:
        final = _vhstack(bcs, scs)

    h, w = final.shape[:2]
    if fig is None:
        fig = plt.figure(num=1, figsize=(10, h / w * 10) if horiz else (w / h * 10, 10))
        fig.clf()

    if ax is None:
        ax = fig.add_axes((0, 0, 1, 1))
    else:
        while len(ax.artists) > 0:
            ax.artists[0].remove()
        while len(ax.texts) > 0:
            ax.texts[0].remove()

    ax.axis("off")
    plt.tight_layout()

    if im is None:
        im = ax.imshow(final)
    else:
        im.set_data(final)

    # ax.plot(*inds.T[::-1], 'o')

    for i, ind in enumerate(inds):
        xy = ind[::-1] - w_inset // 2
        r = Rectangle(xy, w_inset, w_inset, facecolor="none", edgecolor="w")
        ax.add_artist(r)
        ax.text(
            xy[0] - 2,
            xy[1] - 2,
            str(i + 1),
            color="w",
            ha="right",
            va="bottom",
            fontsize=10,
        )
        if horiz:
            ax.text(
                xs[0].shape[1] + 10,
                i * xs[0].shape[0] / n_insets + 10,
                str(i + 1),
                color="w",
                ha="left",
                va="top",
                fontsize=10,
            )
        else:
            ax.text(
                i * xs[0].shape[1] / n_insets + 10,
                xs[0].shape[0] + 10,
                str(i + 1),
                color="w",
                ha="left",
                va="top",
                fontsize=10,
            )

    return fig, ax, im


if __name__ == "__main__":
    matplotlib.use("agg")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--img_raw", type=str, nargs="+", default=None)
    parser.add_argument("--img_cam", type=str, nargs="+", default=None)
    parser.add_argument("-o", "--outdir", type=str, default="cam_insets")
    parser.add_argument("--n_insets", type=int, default=10)
    parser.add_argument("--width_insets", type=int, default=80)
    parser.add_argument("-n", type=int, nargs=2, default=None)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--horiz", action="store_true")
    parser.add_argument("--dry", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.dataset) / args.outdir

    # if data and cams are explicitely given
    if args.img_raw is not None and args.img_cam is not None:
        args.dataset = None
        fxs, fys = sorted(args.img_raw), sorted(args.img_cam)
    # if not, use dataset argument
    else:
        fxs = sorted((Path(args.dataset) / "raws").glob("*.jpg"))
        fys = sorted((Path(args.dataset) / "cam").glob("*.jpg"))

    if len(fxs) != len(fys):
        raise ValueError("different numbers of img and raw files detected!")

    if len(fxs) == 0:
        raise ValueError("Folder {} empty!")

    fxs = fxs[:: args.delta]
    fys = fys[:: args.delta]

    # number of insets to use
    if args.dataset is not None and args.n_insets is None:
        args.n_insets = {"flywing": 15, "fluo01": 10, "fluo02": 10}[args.dataset]

    np.random.seed(42)

    if not args.dry:
        outdir.mkdir(parents=True, exist_ok=True)

    plt.ion()

    # it = tuple(zip(fxs, fys))
    # it = zip(*tuple(it[i:] for i in range(args.frames)))

    itx = tuple(zip(*tuple(fxs[i:] for i in range(args.frames))))
    ity = tuple(zip(*tuple(fys[i:] for i in range(args.frames))))
    it = tuple(zip(itx, ity))

    if args.n is not None:
        it = it[args.n[0] : args.n[1]]
        fys = fys[args.n[0] : args.n[1]]

    # caluclate cam threshold
    if not "th" in locals():
        th = np.percentile(
            np.stack(
                tuple(
                    np.percentile(imread(f)[::2, ::2], 90)
                    for f in tqdm(fys, desc="Get threshold")
                )
            ),
            90,
        )

    for i, (_fxs, _fys) in enumerate(tqdm(it, desc="Make insets")):
        xs = tuple(cached_imread(f, norm=True, togray=True) for f in _fxs)
        ys = tuple(cached_imread(f, norm=True, togray=True) for f in _fys)

        fig, ax, im = cam_insets(
            np.array(xs),
            np.array(ys),
            horiz=args.horiz,
            n_insets=args.n_insets,
            w_inset=args.width_insets,
            # fig=fig,
            # ax=ax,
            # im=im,
            main_frame=(args.frames - 1) // 2,
        )

        if not args.dry:
            fig.savefig(outdir / f"{i:05d}.jpg", dpi=320)
