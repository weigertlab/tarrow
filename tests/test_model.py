from matplotlib import projections
import numpy as np
import torch
import pytest

from tarrow.models import TimeArrowNet


def test_model(classification_head="minimal"):
    model = TimeArrowNet(
        backbone="unet", symmetric=True, classification_head=classification_head
    )

    x = torch.rand((3, 2, 1, 64, 64)).float()

    u = model(x)

    print(f"input:        {x.shape}")
    print(f"output:       {u.shape}")


@pytest.mark.parametrize("projection_head", ["minimal_batchnorm"])
@pytest.mark.parametrize("classification_head", ["minimal"])
@pytest.mark.parametrize("n_frames", [2, 3, 4])
@pytest.mark.parametrize("n_channels", [1, 2, 4])
def test_symmetric(
    projection_head, classification_head, n_frames, n_channels, assert_close=True
):
    model = TimeArrowNet(
        backbone="unet",
        projection_head=projection_head,
        classification_head=classification_head,
        n_input_channels=n_channels,
        n_frames=n_frames,
        symmetric=True,
    )

    x = torch.rand((3, n_frames, n_channels, 64, 64)).float()

    y = model(x)

    for i in range(n_frames):
        x2 = torch.roll(x, i, dims=1)
        y2 = model(x2)
        y2 = torch.roll(y2, -i, dims=1)

        close = torch.allclose(y, y2, atol=1e-5)
        print(x2.shape, y.shape, "  -->   ", close)
        if assert_close:
            assert close


if __name__ == "__main__":
    test_symmetric("minimal_batchnorm", "minimal", 2, 3)
