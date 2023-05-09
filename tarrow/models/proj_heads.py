import logging
from torch import nn

logger = logging.getLogger(__name__)


def _project_heads(
    in_features: int, out_features: int, mode: str = "minimal"
) -> nn.Module:
    if mode == "id":
        layers = nn.Identity()
    elif mode == "linear":
        layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1),
        )
    elif mode == "minimal_batchnorm":
        layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_features),
        )
    elif mode == "three_3x3convs":
        layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.BatchNorm2d(out_features),
        )
    else:
        raise ValueError(f"projection head '{mode}' does not exist.")

    return layers


class ProjectionHead(nn.Module):
    """A head to project the backbone features into the final feature space.
    We can access the features of each layer via hooks.

    input shape: batch, in_features, D0, ..., Dn
    output shape: batch, out_features, D0, ..., Dn

    Args:
        in_features:
            Number of input features per image.
        out_features:
            Number of output features per image.
        mode:
            Projection head architectures. See `_project_heads` for available modes.
    """

    def __init__(self, in_features, out_features, mode="minimal"):
        super().__init__()

        self.out_features = out_features

        logger.debug("classification head: {mode}")
        self.layers = _project_heads(
            in_features=in_features, out_features=out_features, mode=mode
        )

        # Named hooks for all layers
        self.features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        layer_count = 0
        for layer in reversed(list(self.layers.children())):
            if isinstance(layer, (nn.Conv2d, nn.Identity, nn.BatchNorm2d)):
                layer.register_forward_hook(get_activation(layer_count))
                layer_count += 1

    def forward(self, x):
        # Flatten timepoints into the batch dimension
        s_in = x.shape
        x = x.flatten(end_dim=1)
        x = self.layers(x)
        s_out = x.shape
        x = x.reshape(s_in[:2] + (self.out_features,) + s_out[2:])

        return x
