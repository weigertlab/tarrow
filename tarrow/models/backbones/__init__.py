import itertools
import torch
import torchvision

from .simplecnn import SimpleCNN
from .fpn_resnet import FPNResNet
from .lnet_2d import Lnet2d
from .unet_2d import Unet2d
from .resnet import Resnet2d


class PadCropModule(torch.nn.Module):
    """wraps a module such that it can be applied to arbitrary input shapes
    by pad and cropping the input/output

    model = ...

    wrapped_model = PadCropModule(model, div_by = 32, mode= 'reflect')
    """

    def __init__(self, module: torch.nn.Module, div_by: int = 16, mode="reflect"):
        super().__init__()
        self._div_by = div_by
        self._mode = mode
        self._module = module

    def next_valid(self, n: int):
        return ((n + self._div_by - 1) // self._div_by) * self._div_by

    def valid_shape(self, shape):
        return tuple(self.next_valid(s) for s in shape[-2:])

    def pad(self, x, shape):
        """Center-pad x to match spatial dimensions given by shape"""
        assert len(shape) == 2
        pad_total = tuple(a - b for a, b in zip(shape, x.shape[-2:]))
        pad_start, pad_end = tuple(
            zip(*tuple((p // 2, p - (p // 2)) for p in pad_total))
        )
        pad_torch = tuple(itertools.chain(*zip(reversed(pad_start), reversed(pad_end))))
        return torch.nn.functional.pad(x, pad=pad_torch, mode=self._mode)

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""
        assert len(shape) == 2
        x_target_size = x.shape[:-2] + shape
        offset = tuple((a - b) // 2 for a, b in zip(x.shape, x_target_size))
        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))
        return x[slices]

    def forward(self, x: torch.Tensor):
        x_shape = x.shape[-2:]
        x = self.pad(x, self.valid_shape(x_shape))
        x = self._module(x)
        x = self.crop(x, x_shape)
        return x


def get_backbone(backbone="unet", n_input=2):
    if backbone == "unet":
        n_features = 32
        model = Unet2d(
            in_channels=n_input,
            initial_fmaps=32,
            fmap_inc_factor=2,
            downsample_factors=((2, 2), (2, 2), (2, 2)),
            out_channels=n_features,
            batch_norm=True,
            padding=1,
            pad_input=True,
        )
    elif backbone == "lnet":
        n_features = 32
        model = Lnet2d(
            in_channels=n_input,
            initial_fmaps=32,
            fmap_inc_factor=2,
            downsample_factors=((2, 2), (2, 2), (2, 2), (2, 2)),
            upsample_factors=((2, 2), (2, 2), (2, 2)),
            out_channels=n_features,
            batch_norm=True,
            padding=1,
            pad_input=False,
        )
    elif backbone == "unet_16fmaps":
        n_features = 16
        model = Unet2d(
            in_channels=n_input,
            initial_fmaps=16,
            fmap_inc_factor=2,
            downsample_factors=((2, 2), (2, 2), (2, 2)),
            out_channels=n_features,
            batch_norm=True,
            padding=1,
            pad_input=True,
        )
    elif backbone == "resnet_bioimage_wider":
        n_features = 512
        model = Resnet2d(
            in_channels=n_input,
            initial_fmaps=64,
            downsample_factors=(2, 2, 2, 2),
            out_features=n_features,
            fmap_inc_factor=2,
            kernel_sizes=(3, 3),
            padding=1,
            fully_convolutional=True,
        )
    elif backbone == "resnet32":
        n_features = 16
        model = Resnet2d(
            in_channels=n_input,
            initial_fmaps=16,
            downsample_factors=(2, 2, 2),
            out_features=n_features,
            fmap_inc_factor=2,
            kernel_sizes=(3, 3, 3, 3, 3),
            batch_norm=True,
            padding=1,
            fully_convolutional=True,
        )
    elif backbone == "resnet32_32fmaps":
        n_features = 32
        model = Resnet2d(
            in_channels=n_input,
            initial_fmaps=32,
            downsample_factors=(2, 2, 2),
            out_features=n_features,
            fmap_inc_factor=2,
            kernel_sizes=(3, 3, 3, 3, 3),
            batch_norm=True,
            padding=1,
            fully_convolutional=True,
        )
    elif backbone == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=2)
        model.conv1 = torch.nn.Conv2d(n_input, 64, kernel_size=5, stride=2)
        # get everything up to last fc layer
        n_features = model.fc.in_features
        model = torch.nn.Sequential(*(list(model.children())[:-2]))
        model = PadCropModule(model, div_by=32, mode="reflect")
    elif backbone == "simple":
        n_features = 8
        model = SimpleCNN(n_input=n_input, n_output=n_features, n_depth=3)
    elif backbone == "id":
        n_features = n_input
        model = torch.nn.Identity()
    elif backbone == "fpn_resnet":
        n_features = 16
        model = FPNResNet(
            in_channels=n_input,
            out_channels=n_features,
            features=[32, 64, 128, 256],
            stride=2,
        )
    else:
        raise NotImplementedError(backbone)

    return model, n_features
