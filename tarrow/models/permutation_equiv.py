from typing import Literal

import torch
from torch import nn


class DifferenceBlock(nn.Module):
    """
    Just takes the difference of the features.
    """

    def forward(self, x):
        """
        (Batch, n_objects, out_features, D0, ..., Dn)
        """
        assert x.shape[1] == 2

        x1, x2 = torch.split(x, 1, dim=1)
        return torch.cat([x1 - x2, x2 - x1], dim=1)


class BasicPermEquivariantBlock(nn.Module):
    """
    See Zaheer, Manzil, et al. "Deep sets" Neurips 2017 (Eq 23 in the supplements)


    Assumes input to have shape (..., S, in_features) where S is the number of objects (i.e. the set dimension)
    """

    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 64,
        reduce_mode: Literal["mean", "max"] = "mean",
        norm: bool = False,
        activation=nn.LeakyReLU,
        norm_before_act: bool = False,
        **linear_kwargs
    ):
        super().__init__()

        self.L = nn.Linear(in_features, out_features, **linear_kwargs)
        self.G = nn.Linear(in_features, out_features, **linear_kwargs)
        self.act = activation()
        self.norm_before_act = norm_before_act

        if norm:
            self.norm = PermBatchNorm(out_features)
        else:
            self.norm = nn.Identity()

        self.reduce_mode = reduce_mode
        self.reduce_func = {
            "max": lambda x: torch.max(x, -2, keepdims=True)[0],
            "mean": lambda x: torch.mean(x, -2, keepdims=True),
        }[reduce_mode]

    def forward(self, x):
        """
        x.shape -> (..., S, in_features)
        """
        x = self.L(x) + self.reduce_func(self.G(x))

        if self.norm_before_act:
            x = self.norm(x)
            x = self.act(x)
        else:
            x = self.act(x)
            x = self.norm(x)

        return x


class PermBatchNorm(nn.BatchNorm1d):
    """Batchnorm for inputs of shape
    (..., S, C)
    """

    def forward(self, x):
        shape = x.shape
        # reshape to (N, C)
        x = x.flatten(0, -2)
        x = super().forward(x)
        x = x.unflatten(0, shape[:-1])
        return x


class PermEquivariantBlock(nn.Module):
    """A set of stacked BasicPermEquivBlocks

    input  ->  (Batch, n_objects, in_features,  D0, ..., Dn)
    output ->  (Batch, n_objects, out_features, D0, ..., Dn)

    """

    def __init__(
        self,
        n_blocks: int = 2,
        in_features: int = 64,
        out_features: int = 64,
        inter_features: int = 64,
        activation=nn.LeakyReLU,
        norm: bool = False,
        norm_before_act: bool = True,
        bias: bool = True,
        reduce_mode: Literal["mean", "max"] = "mean",
    ):
        super().__init__()

        self.layers = nn.Sequential(
            *tuple(
                BasicPermEquivariantBlock(
                    in_features if i == 0 else inter_features,
                    inter_features if i < n_blocks - 1 else out_features,
                    reduce_mode=reduce_mode,
                    norm=norm,
                    norm_before_act=norm_before_act,
                    activation=activation,
                    bias=bias,
                )
                for i in range(n_blocks)
            )
        )

    def forward(self, x):
        """
        input.shape   ->  (Batch, n_objects, in_features,  D0, ..., Dn)
        output.shape  ->  (Batch, n_objects, out_features, D0, ..., Dn)
        """

        n_batch, n_object = x.shape[:2]

        for permblock in self.layers:
            # permute axis such that perm equiv layers can operate on...
            # (Batch, n_objects, in_features,  D0, ..., Dn) -> (Batch, D0, ..., Dn, n_objects, in_features)
            x = x.moveaxis(1, -1)
            x = x.moveaxis(1, -1)

            x = permblock(x)

            # permute back
            x = x.moveaxis(-1, 1)
            x = x.moveaxis(-1, 1)

        return x
