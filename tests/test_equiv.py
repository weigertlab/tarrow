import pytest
import numpy as np
import torch
from tarrow.models.permutation_equiv import (
    BasicPermEquivariantBlock,
    PermEquivariantBlock,
    DifferenceBlock,
)


def test_perm_equiv_basic():
    for n_set, n_input, n_output in zip((2, 3, 4), (8, 8, 2), (2, 16, 11)):
        shape = tuple(np.random.randint(1, 4, np.random.randint(1, 3)))
        shape = shape + (n_set, n_input)

        x = torch.rand(shape)

        model = BasicPermEquivariantBlock(n_input, n_output)

        y = model(x)

        for i in range(n_set):
            x2 = torch.roll(x, i, dims=-2)
            y2 = model(x2)
            y2 = torch.roll(y2, -i, dims=-2)
            close = torch.allclose(y, y2, atol=1e-5)
            print(x2.shape, close)
            assert y2.shape == x.shape[:-1] + (n_output,)
            assert close


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("n_set", [2, 7])
@pytest.mark.parametrize("in_features", [4, 6])
@pytest.mark.parametrize("out_features", [1, 6])
@pytest.mark.parametrize("spatial", [(2,), (3, 4), (4, 6, 3)])
def test_perm_equiv_head(
    batch, n_set, in_features, out_features, spatial, assert_close=True
):
    shape = (batch, n_set, in_features) + spatial

    x = torch.rand(shape)

    model = PermEquivariantBlock(
        n_blocks=2, in_features=in_features, out_features=out_features
    )

    y = model(x)

    for i in range(n_set):
        x2 = torch.roll(x, i, dims=1)
        y2 = model(x2)
        y2 = torch.roll(y2, -i, dims=1)

        close = torch.allclose(y, y2, atol=1e-5)
        print(x2.shape, y.shape, "  -->   ", close)
        if assert_close:
            assert close


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("in_features", [4, 6])
@pytest.mark.parametrize("out_features", [1, 6])
@pytest.mark.parametrize("spatial", [(2,), (3, 4), (4, 6, 3)])
def test_difference_head(batch, in_features, out_features, spatial, assert_close=True):
    n_set = 2
    shape = (batch, n_set, in_features) + spatial

    x = torch.rand(shape)

    model = DifferenceBlock()

    y = model(x)

    for i in range(n_set):
        x2 = torch.roll(x, i, dims=1)
        y2 = model(x2)
        y2 = torch.roll(y2, -i, dims=1)
        close = torch.allclose(y, y2, atol=1e-5)
        print(x2.shape, y.shape, "  -->   ", close)
        if assert_close:
            assert close


if __name__ == "__main__":
    test_perm_equiv_basic()
