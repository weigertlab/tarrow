import logging
from pathlib import Path
import bisect
from tqdm import tqdm
import tifffile
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from skimage.transform import downscale_local_mean
import skimage
from ..utils import normalize as utils_normalize

logger = logging.getLogger(__name__)


class TarrowDataset(Dataset):
    def __init__(
        self,
        imgs,
        split_start=0,
        split_end=1,
        n_images=None,
        n_frames=2,
        delta_frames=[1],
        subsample=1,
        size=None,
        mode="flip",
        permute=True,
        augmenter=None,
        normalize=None,
        channels=0,
        device="cpu",
        binarize=False,
        random_crop=True,
        reject_background=False,
    ):
        """Returns 2d+time crops. The image sequence is stored in-memory.

        Args:
            imgs:
                Path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays.
            split_start:
                Start point of relative split of image sequence to use.
            split_end:
                End point of relative split of image sequence to use.
            n_images:
                Limit the number of images to use. Useful for debugging.
            n_frames:
                Number of frames in each crop.
            delta_frames:
                Temporal delta(s) between input frames.
            subsample:
                Subsample the input images by this factor.
            size:
                Patch size. If None, use the full image size.
            mode:
                `flip` or `roll` the images along the time axis.
            permute:
                Whether to permute the axes of the images. Set to False for visualizations.
            augmenter:
                Torch transform to apply to the images.
            normalize:
                Image normalization function, applied before croppning. If None, use default percentile-based normalization.
            channels:
                Take the n leading channels from the ones stored in the raw images (leading dimension). 0 means there is no channel dimension in raw files.
            device:
                Where to store the precomputed crops.
            binarize:
                Binarize the input images. Should only be used for images stored in integer format.
            random_crop:
                If `True`, crop random patches in spatial dimensions. If `False`, center-crop the images (e.g. for visualization).
            reject_background:
                help="Set to `True` to heuristically reject background patches.
        """

        super().__init__()

        self._split_start = split_start
        self._split_end = split_end
        self._n_images = n_images
        self._n_frames = n_frames
        self._delta_frames = delta_frames
        self._subsample = subsample

        assert mode in ["flip", "roll"]
        self._mode = mode

        self._permute = permute
        self._channels = channels
        self._device = device
        self._augmenter = augmenter
        if self._augmenter is not None:
            self._augmenter.to(device)

        if isinstance(imgs, (str, Path)):
            imgs = self._load(
                path=imgs,
                split_start=split_start,
                split_end=split_end,
                n_images=n_images,
            )
        elif isinstance(imgs, (tuple, list, np.ndarray)) and isinstance(
            imgs[0], np.ndarray
        ):
            imgs = np.asarray(imgs)[:n_images]
        else:
            raise ValueError(
                f"Cannot form a dataset from {imgs}. "
                "Input should be either a path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays."
            )

        if self._channels == 0:
            imgs = np.expand_dims(imgs, 1)
        else:
            imgs = imgs[:, : self._channels, ...]

        assert imgs.shape[1] == 1

        if binarize:
            logger.debug("Binarize images")
            imgs = (imgs > 0).astype(np.float32)
        else:
            logger.debug("Normalize images")
            if normalize is None:
                imgs = self._default_normalize(imgs)
            else:
                imgs = normalize(imgs)

        imgs = torch.as_tensor(imgs)

        if not isinstance(subsample, int) or subsample < 1:
            raise NotImplementedError(
                "Spatial subsampling only implemented for positive integer values."
            )
        if subsample > 1:
            factors = (1,) + (subsample,) * (imgs.dim() - 1)
            full_size = imgs[0].shape
            imgs = downscale_local_mean(imgs, factors)
            logger.debug(f"Subsampled from {full_size} to {imgs[0].shape}")

        if size is None:
            self._size = imgs[0, 0].shape
        else:
            # assert np.all(
            # np.array(size) <= np.array(imgs[0, 0].shape)
            # ), f"{size=} {imgs[0,0].shape=}"
            # self._size = size

            self._size = tuple(min(a, b) for a, b in zip(size, imgs[0, 0].shape))

        if random_crop:
            if reject_background:
                self._crop = self._reject_background()
            else:
                self._crop = transforms.RandomCrop(
                    self._size,
                    padding_mode="reflect",
                    pad_if_needed=True,
                )
        else:
            self._crop = transforms.CenterCrop(self._size)

        if imgs.ndim != 4:  # T, C, X, Y
            raise NotImplementedError(
                f"only 2D timelapses supported (total image shape: {imgs.shape})"
            )
        min_number = max(self._delta_frames) * (n_frames - 1) + 1
        if len(imgs) < min_number:
            raise ValueError(f"imgs should contain at last {min_number} elements")
        if len(imgs.shape[2:]) != len(self._size):
            raise ValueError(
                f"incompatible shapes between images and size last {n_frames} elements"
            )

        # Precompute the time slices
        self._imgs_sequences = []
        for delta in self._delta_frames:
            n, k = self._n_frames, delta
            logger.debug(f"Creating delta {delta} crops")
            tslices = tuple(
                slice(i, i + k * (n - 1) + 1, k) for i in range(len(imgs) - (n - 1) * k)
            )
            imgs_sequences = [torch.as_tensor(imgs[ss]) for ss in tslices]
            self._imgs_sequences.extend(imgs_sequences)

        self._crops_per_image = max(
            1, int(np.prod(imgs.shape[1:3]) / np.prod(self._size))
        )

    def _reject_background(self, threshold=0.02, max_iterations=10):
        rc = transforms.RandomCrop(
            self._size,
            padding_mode="reflect",
            pad_if_needed=True,
        )

        def smoother(img):
            img = skimage.util.img_as_ubyte(img.squeeze(1).numpy().clip(-1, 1))
            img = skimage.filters.rank.median(
                img, footprint=np.ones((self._n_frames, 3, 3))
            )
            return torch.as_tensor(skimage.util.img_as_float32(img)).unsqueeze(1)

        def crop(x):
            with torch.no_grad():
                for i in range(max_iterations):
                    out = rc(x)
                    mask = smoother(out)
                    if mask.std() > threshold:
                        return out
                    logger.debug(f"Reject {i}")
                return out

        return crop

    def _default_normalize(self, imgs):
        """Default normalization.

        Normalizes each image separately. Can be overwritten in subclasses.

        Args:
            imgs: List of images or ndarray.

        Returns:
            ndarray

        """
        imgs_norm = []
        for img in tqdm(imgs, desc="normalizing images", leave=False):
            imgs_norm.append(utils_normalize(img, subsample=8))
        return np.stack(imgs_norm)

    def _load(self, path, split_start, split_end, n_images=None):
        """Loads image from disk into CPU memory.

        Can be overwritten in subclass for particular datasets.

        Args:
            path(``str``):
                Dataset directory.
            split_start(``float``):
                Use only images after this fraction of the dataset. Defaults to 0.
            split_end(``float``):
                Use only images before this fraction of the dataset. Defaults to 1.
            n_images(``int``):
                Limit number of used images. Set to ``None`` to use all avaible images.

        Returns:
            Numpy array of shape(imgs, dim0, dim1, ... , dimN).
        """

        assert split_start >= 0
        assert split_end <= 1

        inp = Path(path).expanduser()

        if inp.is_dir():
            suffixes = ("png", "jpg", "tif", "tiff")
            for s in suffixes:
                fnames = sorted(Path(inp).glob(f"*.{s}"))
                if len(fnames) > 0:
                    break
            if len(fnames) == 0:
                raise ValueError(f"Could not find ay images in {inp}")

            fnames = fnames[:n_images]
            imgs = self._load_image_folder(fnames, split_start, split_end)

        elif inp.suffix == ".tif":
            logger.info(f"Loading {inp}")
            imgs = tifffile.imread(str(inp))
            logger.info("Done")
            assert imgs.ndim == 3
            imgs = imgs[int(len(imgs) * split_start) : int(len(imgs) * split_end)]
            imgs = imgs[:n_images]
        else:
            raise ValueError(
                (
                    f"Cannot form a dataset from {inp}. "
                    "Input should be either a path to a sequence of 2d images, a single 2d+time image, or a list of 2d np.ndarrays."
                )
            )

        return imgs

    def _load_image_folder(
        self,
        fnames,
        split_start: float,
        split_end: float,
    ) -> np.ndarray:
        idx_start = int(len(fnames) * split_start)
        idx_end = int(len(fnames) * split_end)
        fnames = fnames[idx_start:idx_end]

        logger.info(f"Load images {idx_start}-{idx_end}")
        imgs = []
        for f in tqdm(fnames, leave=False, desc="loading images"):
            f = Path(f)
            if f.suffix in (".tif", ".TIFF", ".tiff"):
                x = tifffile.imread(f)
            elif f.suffix in (".png", ".jpg", ".jpeg"):
                x = imageio.imread(f)
                if x.ndim == 3:
                    x = np.moveaxis(x[..., :3], -1, 0)
            else:
                continue
            x = np.squeeze(x)
            imgs.append(x)
        imgs = np.stack(imgs)
        return imgs

    def __len__(self):
        return len(self._imgs_sequences)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return list(self[_idx] for _idx in idx)

        x = self._imgs_sequences[idx]

        x = self._crop(x)

        if self._permute:
            if self._mode == "flip":
                label = torch.randint(0, 2, (1,))[0]
                if label == 1:
                    x = torch.flip(x, dims=(0,))
            elif self._mode == "roll":
                label = torch.randint(0, self._n_frames, (1,))[0]
                x = torch.roll(x, label.item(), dims=(0,))
            else:
                raise ValueError()
        else:
            label = torch.tensor(0, dtype=torch.long)

        x, label = x.to(self._device), label.to(self._device)

        if self._augmenter is not None:
            x = self._augmenter(x)

        return x, label


class ConcatDatasetWithIndex(ConcatDataset):
    """Additionally returns index"""

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], idx
