import logging
from typing import Tuple
import torch
from torch import nn
import torchvision

from .backbones import Unet2d
from .permutation_equiv import (
    DifferenceBlock,
    PermEquivariantBlock,
)

logger = logging.getLogger(__name__)


def _classification_heads(
    in_features: int,
    out_features: int,
    n_classes: int,
    mode: str = "minimal",
) -> Tuple[nn.Module, nn.Module]:
    """Standard classification heads.
    Returns layers and head module.

    Input shape: B, in_features, D0, ..., Dn
    Layers output shape: B, out_features, D0, ..., Dn
    Head output shape: B, n_classes
    """

    if mode == "unet":
        layers = nn.Sequential(
            Unet2d(
                in_channels=in_features,
                initial_fmaps=32,
                fmap_inc_factor=2,
                downsample_factors=((2, 2), (2, 2), (2, 2)),
                out_channels=out_features,
                batch_norm=True,
                padding=1,
                pad_input=True,
            ),
            nn.Conv2d(out_features, n_classes, kernel_size=1, bias=False),
        )
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    elif mode == "resnet18":
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(
            in_features,
            64,
            kernel_size=5,
            stride=2,
        )
        layers = nn.Sequential(
            *(list(resnet.children())[:-2]),
            nn.Conv2d(512, n_classes, kernel_size=1, bias=False),
        )
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    elif mode == "minimal":
        layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1),
            nn.LeakyReLU(),
            nn.Conv2d(out_features, out_features, 1),
        )
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_features, n_classes, bias=False),
        )
    else:
        raise ValueError(f"Classification head '{mode}' does not exist.")

    return layers, head


def _classification_heads_symmetric(
    in_features: int,
    out_features: int,
    mode: str = "minimal",
) -> Tuple[nn.Module, nn.Module]:
    """Symmetric classification heads
    Returns layers and head module.

    Input shape: B, in_features, D0, ..., Dn
    Layers output shape: B, out_features, D0, ..., Dn
    Head output shape: B, n_classes
    """

    if mode == "difference":
        layers = nn.Sequential(
            DifferenceBlock(),
            PermEquivariantBlock(
                n_blocks=1,
                in_features=in_features,
                out_features=1,
                activation=torch.nn.Identity,
                norm=False,
            ),
            nn.Flatten(1, 2),
        )
        head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    elif mode == "linear":
        layers = nn.Sequential(
            PermEquivariantBlock(
                n_blocks=1,
                in_features=in_features,
                out_features=out_features,
                activation=torch.nn.Identity,
                norm_before_act=True,
            ),
        )
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            PermEquivariantBlock(
                n_blocks=1,
                in_features=out_features,
                out_features=1,
                activation=torch.nn.Identity,
                norm=False,
            ),
            # remove now obsolete time dimension
            nn.Flatten(),
        )

    elif mode == "minimal":
        layers = nn.Sequential(
            PermEquivariantBlock(
                n_blocks=1,
                in_features=in_features,
                out_features=out_features,
                activation=torch.nn.LeakyReLU,
                norm=True,
                norm_before_act=True,
            ),
            PermEquivariantBlock(
                n_blocks=1,
                in_features=out_features,
                out_features=out_features,
                activation=torch.nn.LeakyReLU,
                norm=True,
                norm_before_act=True,
            ),
        )
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            PermEquivariantBlock(
                n_blocks=1,
                in_features=out_features,
                out_features=1,
                activation=torch.nn.Identity,
                norm=False,
            ),
            # remove now obsolete time dimension
            nn.Flatten(),
        )
    else:
        raise ValueError(f"Symmetric classification head '{mode}' does not exist.")

    return layers, head


class ClassificationHead(nn.Module):
    """Gets individual dense feature maps for each image as an input.
    Classifies the input into `n_classes` classes.

    Args:
        in_features:
            Number of input features per image.
        n_frames:
            Number of timesteps per sample.
        out_features:
            Number of hidden features per image.
        n_classes:
            Number of output classes (2 for flipping mode).
        mode:
            Classification head architecture.
        symmetric:
            Whether to use a permutation-equivariant classification head.
    """

    def __init__(
        self,
        in_features,
        n_frames,
        out_features,
        n_classes,
        mode="minimal",
        symmetric=False,
    ):
        super().__init__()

        logger.debug("Classification head: {mode}")

        self.symmetric = symmetric

        if self.symmetric:
            assert n_frames == n_classes
            self.layers, self.head = _classification_heads_symmetric(
                in_features=in_features,
                out_features=out_features,
                mode=mode,
            )

        else:
            # as we concatenate `n_frames`, `in_features` is multiplied by that amount
            self.layers, self.head = _classification_heads(
                in_features=n_frames * in_features,
                out_features=out_features,
                n_classes=n_classes,
                mode=mode,
            )

    def forward(self, x):
        """
        Args:
            x: Tensor of size (Batch, Timepoints, in_features, D0, ..., Dn)

        Returns:
            Tensor of size (Batch, n_classes)
        """

        if not self.symmetric:
            # Concat all the timepoints in the channel dimension
            x = x.flatten(start_dim=1, end_dim=2)

        x = self.layers(x)
        x = self.head(x)

        return x
