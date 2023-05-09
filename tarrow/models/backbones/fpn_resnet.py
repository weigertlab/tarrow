from abc import abstractmethod
import itertools
import torch
from torch import Tensor, nn
from typing import Callable, OrderedDict, List, Optional
import torchvision
from torchvision.models.resnet import conv3x3


def crop_to_shape(x, shape):
    assert len(shape) == 2
    assert all(s2 <= s1 for s1, s2 in zip(x.shape[-2:], shape))
    slices = tuple(
        slice((s1 - s2) // 2, s2 + (s1 - s2) // 2)
        for s1, s2 in zip(x.shape[-2:], shape)
    )
    slices = (slice(None),) * (len(x.shape) - 2) + slices
    return x[slices]


def pad_to_shape(x, shape, padding_mode="reflect"):
    assert len(shape) == 2
    assert all(s1 <= s2 for s1, s2 in zip(x.shape[-2:], shape))

    pad_left = tuple((s2 - s1) // 2 for s1, s2 in zip(x.shape[-2:], shape))

    pad_right = tuple((s2 - s1) - (s2 - s1) // 2 for s1, s2 in zip(x.shape[-2:], shape))

    # reverse axis order and interleave padding tuples
    # for torch.nn.functional.pad
    pad = tuple(itertools.chain(*zip(reversed(pad_left), reversed(pad_right))))
    return torch.nn.functional.pad(input=x, pad=pad, mode=padding_mode)


class PadCropModule(object):
    @abstractmethod
    def forward_impl(self, x, **kwargs):
        pass

    @abstractmethod
    def _divby(self):
        pass

    def _pad_shape(self, shape):
        assert len(shape) == 2
        w = self._divby()
        return tuple((s + (w - s % w) % w) for s in shape)

    def _preproc(self, x):
        shape = self._pad_shape(x.shape[-2:])
        self._orig_shape = x.shape[-2:]
        if shape == self._orig_shape:
            return x
        else:
            return pad_to_shape(x, shape)

    def _postproc(self, x):
        if x.shape[-2:] == self._orig_shape:
            return x
        else:
            return crop_to_shape(x, self._orig_shape)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation=nn.LeakyReLU,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = activation(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBlock(nn.Module, PadCropModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 16,
        features: List[int] = [32, 64, 128],
        blocks_per_layer: int = 2,
        stride: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation=nn.LeakyReLU,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.stride = stride
        self.groups = groups
        self.base_width = width_per_group

        layers = [
            self._make_layer(
                in_channels,
                features[0],
                blocks_per_layer,
                stride=1,
                activation=activation,
            )
        ]
        for i, (n1, n2) in enumerate(zip(features, features[1:])):
            layers.append(
                self._make_layer(
                    n1, n2, blocks_per_layer, stride=self.stride, activation=activation
                )
            )

        self.layers = nn.ModuleList(layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.final = nn.Conv2d(features[-1], out_channels, 1, padding=0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        in_planes: int,
        out_planes: int,
        n_blocks: int,
        stride: int = 1,
        dilate: bool = False,
        activation=nn.LeakyReLU,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                norm_layer(out_planes),
            )

        layers = []
        layers.append(
            BasicBlock(
                in_planes,
                out_planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                activation=activation,
            )
        )

        for _ in range(1, n_blocks):
            layers.append(
                BasicBlock(
                    out_planes,
                    out_planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation,
                )
            )

        return nn.Sequential(*layers)

    def _divby(self):
        return self.stride ** (len(self.layers) - 1)

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        # See note [TorchScript super()]

        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = self.final(x)

        if return_features:
            return feats
        else:
            return x


class FPNResNet(nn.Module, PadCropModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 16,
        features: List[int] = [32, 64, 128],
        blocks_per_layer: int = 2,
        pre_stride=2,
        stride=2,
        fuse_final=True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation=nn.LeakyReLU,
    ) -> nn.Module:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.fuse_final = fuse_final
        self.act = activation(inplace=True)
        self.pre_stride = pre_stride

        self.first = nn.Sequential(
            nn.Conv2d(
                in_channels, features[0], 3, padding=1, stride=pre_stride, bias=False
            ),
            norm_layer(features[0]),
            activation(inplace=True),
        )

        self.backbone = ResNetBlock(
            in_channels=features[0],
            out_channels=out_channels,
            features=features,
            blocks_per_layer=blocks_per_layer,
            stride=stride,
            norm_layer=norm_layer,
        )

        self.fpn = torchvision.ops.FeaturePyramidNetwork(features, out_channels)
        self.upsamples = nn.ModuleList(
            nn.Upsample(
                scale_factor=pre_stride * stride**i,
                mode="bilinear",
                align_corners=True,
            )
            for i in range(len(features))
        )

        self.final = nn.Sequential(
            activation(inplace=True),
            nn.Conv2d(len(features) * out_channels, out_channels, 3, padding=1),
        )

    def _divby(self):
        return self.pre_stride * self.backbone._divby()

    def forward(self, x):
        x = self._preproc(x)

        x = self.first(x)
        x = self.backbone(x, return_features=True)
        xs = OrderedDict((f"feat{i}", _x) for i, _x in enumerate(x))
        xs = self.fpn(xs).values()
        xs = tuple(u(x) for x, u in zip(xs, self.upsamples))
        xs = torch.cat(xs, axis=1)

        if self.fuse_final:
            x = self.final(xs)
        else:
            x = xs

        x = self._postproc(x)
        return x


if __name__ == "__main__":
    x = torch.rand(2, 1, 128, 128)

    model = ResNetBlock(1, 32)
    # x = x.cuda()

    model = FPNResNet(1, 32, features=[32, 64, 128, 256], stride=2)

    # model.eval()
    # model.to('cuda')

    y = model(x)

    print(y.shape)
