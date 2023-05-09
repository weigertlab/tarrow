import torch
from .augmentations import (
    RandomFlipRot,
    RandomIntensity,
    RandomRotate,
    RandomElastic,
    RandomNoise,
    RandomAffine,
)


def get_augmenter(augment_id: int) -> torch.nn.Module:
    """Augmentations for tensor of (B),T,C,H,W.

    All axis parameters get mapped to only T,C,H,W.

    Parameters
    ----------
    augment_id : int
        augment_id

    Returns
    -------
    torch.nn.Module

    """
    if augment_id == 0:
        aug = torch.nn.Sequential()
    elif augment_id == 1:  # fliprot
        aug = torch.nn.Sequential(RandomFlipRot())
    elif augment_id == 2:  # fliprot + pixelwise
        prob = 0.5
        aug = torch.nn.Sequential(
            RandomFlipRot(),
            RandomNoise(probability=prob),
            # frame by frame intensity shift
            RandomIntensity(
                shift=(-0.05, 0.05), scale=(9 / 10, 10 / 9), axis=0, probability=prob
            ),
            # global intensity shift
            RandomIntensity(shift=(-0.1, 0.1), scale=(4 / 5, 5 / 4), probability=prob),
        )
    elif augment_id == 3:  # fliprot + rotation + pixelwise
        prob = 0.5
        aug = torch.nn.Sequential(
            RandomFlipRot(),
            RandomRotate(probability=prob),
            RandomNoise(probability=prob),
            # frame by frame intensity shift
            RandomIntensity(
                shift=(-0.05, 0.05), scale=(9 / 10, 10 / 9), axis=0, probability=prob
            ),
            # global intensity shift
            RandomIntensity(shift=(-0.1, 0.1), scale=(4 / 5, 5 / 4), probability=prob),
        )
    elif augment_id == 4:  # fliprot + rotation + scaling + deformation + pixelwise
        prob = 0.5
        aug = torch.nn.Sequential(
            RandomFlipRot(),
            RandomRotate(probability=prob),
            RandomAffine(axis=0, scale=(9 / 10, 10 / 9), probability=prob),
            RandomElastic(probability=prob),
            RandomNoise(probability=prob),
            # frame by frame intensity shift
            RandomIntensity(
                shift=(-0.05, 0.05), scale=(9 / 10, 10 / 9), axis=0, probability=prob
            ),
            # global intensity shift
            RandomIntensity(shift=(-0.1, 0.1), scale=(4 / 5, 5 / 4), probability=prob),
        )
    elif (
        augment_id == 5
    ):  # fliprot + rotation + scaling + translation + deformation + pixelwise
        prob = 0.5
        aug = torch.nn.Sequential(
            RandomFlipRot(),
            RandomAffine(
                degrees=180,
                scale=(4 / 5, 5 / 4),
                probability=prob,
            ),
            RandomAffine(
                axis=0,
                translate=(0.03, 0.03),
                scale=(9 / 10, 10 / 9),
                probability=prob,
            ),
            RandomElastic(probability=prob),
            RandomNoise(probability=prob),
            # frame by frame intensity shift
            RandomIntensity(
                shift=(-0.05, 0.05), scale=(9 / 10, 10 / 9), axis=0, probability=prob
            ),
            # global intensity shift
            RandomIntensity(shift=(-0.1, 0.1), scale=(4 / 5, 5 / 4), probability=prob),
        )
    else:
        raise ValueError(f"{augment_id=}")

    return aug
