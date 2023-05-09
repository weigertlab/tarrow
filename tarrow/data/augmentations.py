"""
Fast torch versions of batched augmentation functions to be
directly applied to tarrow batched images on the GPU
"""

from abc import abstractmethod, ABC
from numbers import Number
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def tensor_grid(shape):
    """creates a default coordinate meshgrid with values[-1,1]"""
    u0 = np.stack(
        np.meshgrid(
            np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij"
        ),
        axis=-1,
    )
    # flow vector in pytorch is in xyz order
    u0 = u0[:, :, ::-1].copy()
    return torch.tensor(u0).float().unsqueeze(0)


def torch_uniform(low, high, shape):
    x = torch.rand(shape)
    x = low + (high - low) * x
    return x


class BaseTransform(torch.nn.Module, ABC):
    def __init__(self, probability: float = 1.0):
        super().__init__()
        self._probability = probability

    @abstractmethod
    def forward_impl(self, x: torch.Tensor):
        pass

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            p = torch.rand(1).item()
            if p <= self._probability:
                return self.forward_impl(x)
            else:
                return x
        elif x.ndim == 5:
            p = torch.rand(len(x))
            return torch.stack(
                tuple(
                    self.forward_impl(_x) if _p <= self._probability else _x
                    for _x, _p in zip(x, p)
                )
            )
        else:
            raise ValueError("Transform assumes 4D or 5D data! (B),T,C,H,W")


class RandomFlipRot(BaseTransform):
    def forward_impl(self, x):
        dims = (2 + np.where(np.random.randint(0, 2, 2) == 0)[0]).tolist()
        return torch.flip(x, dims)


class RandomRotate(BaseTransform):
    def __init__(self, mode="bilinear", padding_mode="reflection", probability=1.0):
        super().__init__(probability=probability)
        self._mode = mode
        self._padding_mode = padding_mode
        self._grid_cache = dict()

    def _get_grid(self, n_frames, shape):
        # if shape in self._grid_cache:
        # return self._grid_cache[shape]
        # else:
        # grid = tensor_grid(shape)
        # grid = grid.expand((n_frames,) + shape + (2,)).unsqueeze(-1)
        # self._grid_cache[shape] = grid
        # return grid
        grid = tensor_grid(shape)
        grid = grid.expand((n_frames,) + shape + (2,)).unsqueeze(-1)
        return grid

    def forward_impl(self, x):
        # x -> TCHW

        n_frames = len(x)
        spatial_shape = x.shape[-2:]

        w = 2 * np.pi * torch.rand(1)
        w = w.broadcast_to((n_frames, 1, 1))

        M_rot = torch.stack(
            (
                torch.stack((torch.cos(w), -torch.sin(w)), -1),
                torch.stack((torch.sin(w), torch.cos(w)), -1),
            ),
            -1,
        )
        grid = self._get_grid(n_frames, spatial_shape)
        grid = torch.matmul(M_rot, grid)
        grid = grid.squeeze(-1)
        grid = grid.broadcast_to((n_frames,) + spatial_shape + (2,))
        grid = grid.to(x.device)

        return F.grid_sample(
            x,
            grid,
            mode=self._mode,
            padding_mode=self._padding_mode,
            align_corners=True,
        )


class RandomElastic(BaseTransform):
    def __init__(
        self,
        grid=(5, 5),
        amount=10,
        mode="bilinear",
        padding_mode="reflection",
        probability=1.0,
        axis=None,
    ):
        super().__init__(probability=probability)
        self._grid = grid
        self._amount = amount
        self._mode = mode
        self._padding_mode = padding_mode
        self._axis = axis

    def forward_impl(self, x):
        """x -> T,C,H,W"""

        align_corners = self._mode != "nearest"

        spatial_shape = x.shape[-2:]
        amount_normed = self._amount / torch.tensor(spatial_shape)

        if self._axis is None:
            amount = torch_uniform(-1, 1, (1,) + self._grid + (2,)) * amount_normed
            amount = amount.broadcast_to((len(x),) + self._grid + (2,))
        elif self._axis == 0:
            amount = (
                torch_uniform(-1, 1, (x.shape[0],) + self._grid + (2,)) * amount_normed
            )
        else:
            raise ValueError()

        amount = F.interpolate(
            amount.moveaxis(-1, 1), spatial_shape, align_corners=True, mode="bilinear"
        ).moveaxis(1, -1)

        grid = tensor_grid(spatial_shape)

        grid = grid + amount

        grid = grid.to(x.device)

        return F.grid_sample(
            x,
            grid,
            mode=self._mode,
            padding_mode=self._padding_mode,
            align_corners=align_corners,
        )


class RandomAffine(BaseTransform):
    def __init__(
        self,
        degrees=0,
        translate=None,
        scale=None,
        shear=None,
        interpolation="bilinear",
        fill=0,
        axis=None,
        probability: float = 1.0,
    ):
        super().__init__(probability=probability)
        if axis is None or isinstance(axis, int):
            self._axis = axis
        else:
            raise ValueError()

        interpolation = {
            "nearest": torchvision.transforms.functional.InterpolationMode.NEAREST,
            "bilinear": torchvision.transforms.functional.InterpolationMode.BILINEAR,
        }[interpolation]

        self.transform = torchvision.transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
        )

    def forward_impl(self, x):
        """x -> TCHW"""
        if self._axis is None:
            return self.transform(x)
        else:
            return torch.concat(
                tuple(self.transform(_x) for _x in torch.split(x, 1, dim=self._axis))
            )


class RandomIntensity(BaseTransform):
    def __init__(
        self, shift=(-0.05, 0.05), scale=(0.9, 1.2), axis=None, probability=1.0
    ):
        """Use a different random shift/scale for every element of axis (just global one if axis is None)."""
        super().__init__(probability=probability)
        self._scale = scale
        self._shift = shift
        if axis is None:
            self._axis = ()
        elif isinstance(axis, Number):
            self._axis = (axis,)
        elif isinstance(axis, tuple):
            self._axis = axis
        else:
            raise ValueError()

    def forward_impl(self, x):
        ax = tuple(range(x.ndim))
        ax = tuple(ax[a] for a in self._axis)
        shape = tuple(x.shape[i] if i in ax else 1 for i in range(x.ndim))
        scale = torch_uniform(*self._scale, shape=shape).to(x.device)
        shift = torch_uniform(*self._shift, shape=shape).to(x.device)
        x = scale * x + shift
        return x


class RandomNoise(BaseTransform):
    def __init__(self, sigma=0.05, probability=1.0):
        super().__init__(probability=probability)
        self._sigma = sigma
        super().__init__()

    def forward_impl(self, x):
        x = x + torch.rand(1)[0] * self._sigma * torch.rand(*x.shape).to(x.device)
        return x


if __name__ == "__main__":
    import tarrow

    device, n_gpus = tarrow.utils.set_device(0)

    x = 0.4 * torch.rand(2, 1, 128, 128)
    x[0, :, 40:60, 40:60] = 1
    x[1, :, 40:60, 40:60] = 2

    x = x.to(device)

    # y = RandomFlipRot()(x)
    # y = RandomRotate(probability=.8)(x)

    # y = RandomElastic(mode='nearest', probability=1)(x)
    # # y = RandomRotate()(x)
    # y = RandomIntensity(scale=(1,4), axis=0)(x)
    # y = RandomAffine(independent=True, scale=(0.9, 1.1))(x)

    # y = RandomElastic()(x)
    # y = RandomNoise(sigma=1)(x)

    y = RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        # scale=(0.5, 2.0),
        axis=0,
    )(x)

    assert x.shape == y.shape
