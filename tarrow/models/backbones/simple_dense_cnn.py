import torch
from torch import nn
from typing import Callable, Any, Optional, List


class BasicConv(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        padding_mode: Optional[str] = "replicate",
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class SimpleDenseCNN(nn.Module):
    def __init__(self, n_input=2, n_output=1, n_depth=3, n_features=32):
        super().__init__()

        self.conv1 = BasicConv(n_input, n_features, stride=1, padding=1)

        self.convs = nn.Sequential(
            *tuple(
                BasicConv(n_features, n_features, stride=1, padding=1)
                for _ in range(n_depth)
            )
        )

        self.final = BasicConv(n_features, n_output, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.convs(x)
        x = self.final(x)
        return x
