from typing import Sequence
import logging
import platform
from pathlib import Path
from datetime import datetime
import yaml
import git
import configargparse
import time

import torch
from torch.utils.data import ConcatDataset, Subset, Dataset


import tarrow
from tarrow.models import TimeArrowNet
from tarrow.data import TarrowDataset, get_augmenter
from tarrow.visualizations import create_visuals


logging.basicConfig(
    format="%(filename)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_argparser():
    p = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )

    p.add(
        "-c",
        "--config",
        is_config_file=True,
        help="Config file path (other given arguments will superseed this).",
    )
    p.add("--name", type=str, default=None, help="Name of the training run.")
    p.add(
        "--input_train",
        type=str,
        nargs="+",
        required=True,
        help="Input training data. Can be 2D+time images, or directories with time sequences of 2d images.",
    )
    p.add(
        "--input_val",
        type=str,
        nargs="*",
        default=None,
        help="Same as `--input_train`. If not given, `--input_train` is used for validation.",
    )
    p.add("--read_recursion_level", type=int, default=0)
    p.add(
        "--split_train",
        type=float,
        nargs=2,
        action="append",
        required=True,
        help="Relative split of training data as (start, end).",
    )
    p.add(
        "--split_val",
        type=float,
        nargs="+",
        action="append",
        required=True,
        help="Same as `--split_val`.",
    )
    p.add("-e", "--epochs", type=int, default=200)
    p.add("--seed", type=int, default=42)
    p.add("--backbone", type=str, default="unet")
    p.add("--projhead", default="minimal_batchnorm")
    p.add("--classhead", default="minimal")
    p.add(
        "--perm_equiv",
        type=tarrow.utils.str2bool,
        default=True,
        help="Whether to use permutation equivariant prediction head.",
    )
    p.add(
        "--features",
        type=int,
        default=32,
        help="Dimesionality of the dense representations.",
    )
    p.add(
        "--n_images",
        type=int,
        default=None,
        help="Limit the number of images to use. Useful for debugging.",
    )
    p.add(
        "-o",
        "--outdir",
        type=str,
        default="runs",
        help="Save models and tensorboard here.",
    )
    p.add("--size", type=int, default=96, help="Patch size for training.")
    p.add(
        "--cam_size",
        type=int,
        default=None,
        help="Patch size for CAM visualization. If not given, full images are used.",
    )
    p.add("--batchsize", type=int, default=128)
    p.add("--train_samples_per_epoch", type=int, default=100000)
    p.add("--val_samples_per_epoch", type=int, default=10000)
    p.add(
        "--channels",
        type=int,
        default=0,
        help="Number of channels in the input images. Set to 0 for images do not have a explicit channel dimension.",
    )
    p.add(
        "--reject_background",
        type=tarrow.utils.str2bool,
        default=False,
        help="Set to `True` to heuristically reject background patches during training.",
    )
    p.add(
        "--cam_subsampling",
        type=int,
        default=3,
        help="Number of time frames with periodic CAM visualization.",
    )
    p.add(
        "--write_final_cams",
        type=tarrow.utils.str2bool,
        default=False,
        help="Write out CAMs of validation datasets after training is finished.",
    )
    p.add(
        "--augment",
        type=int,
        default=5,
        help="Level of data augmentation from 0 (no augmentation) to 5 (strong augmentation).",
    )
    p.add(
        "--subsample",
        type=int,
        default=1,
        help="Subsample the input images by this factor.",
    )
    p.add(
        "--delta",
        type=int,
        nargs="+",
        default=[1],
        help="Temporal delta(s) between input frames.",
    )
    p.add(
        "--frames",
        type=int,
        default=2,
        help="Number of input frames for each training sample.",
    )
    p.add("--lr", type=float, default=1e-4)
    p.add("--lr_scheduler", default="cyclic")
    p.add("--lr_patience", type=int, default=50)
    p.add("--ndim", type=int, default=2)
    p.add(
        "--binarize",
        action="store_true",
        help="Binarize the input images. Should only be used for images stored in integer format.",
    )
    p.add(
        "--decor_loss",
        type=float,
        default=0.01,
        help="Relative weighting of the decorrelation loss.",
    )
    p.add("--save_checkpoint_every", type=int, default=25)
    p.add("--num_workers", type=int, default=8, help="Number of CPU workers.")
    p.add(
        "--gpu",
        "-g",
        type=str,
        default="0",
        help="GPUs to use. Can be a single integer, a comma-separated list of integers, or an interval `a-b`, or 'cpu'.",
    )
    p.add("--tensorboard", type=tarrow.utils.str2bool, default=True)
    p.add(
        "--visual_dataset_frequency",
        type=int,
        default=10,
        help="Save attribution maps to tensorboard every n-th epoch.",
    )
    p.add(
        "--timestamp",
        action="store_true",
        help="Prepend output directory name with timestamp.",
    )

    return p


def _get_paths_recursive(paths: Sequence[str], level: int):
    input_rec = paths
    for i in range(level):
        new_inps = []
        for i in input_rec:
            if Path(i).is_dir():
                children = [
                    x for x in Path(i).iterdir() if x.is_dir() or x.suffix == ".tif"
                ]
                new_inps.extend(children)
            if Path(i).suffix == ".tif":
                new_inps.append(Path(i))
        input_rec = new_inps
    return input_rec


def _build_dataset(
    imgs,
    split,
    size,
    args,
    n_frames,
    delta_frames,
    augmenter=None,
    permute=True,
    random_crop=True,
    reject_background=False,
):
    return TarrowDataset(
        imgs=imgs,
        split_start=split[0],
        split_end=split[1],
        n_images=args.n_images,
        n_frames=n_frames,
        delta_frames=delta_frames,
        subsample=args.subsample,
        size=size,
        mode="flip",
        permute=permute,
        augmenter=augmenter,
        device="cpu",
        channels=args.channels,
        binarize=args.binarize,
        random_crop=random_crop,
        reject_background=reject_background,
    )


def _subset(data: Dataset, split=(0, 1.0)):
    low, high = int(len(data) * split[0]), int(len(data) * split[1])
    return Subset(data, range(low, high))


def _create_loader(dataset, args, num_samples, num_workers, idx=None, sequential=False):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(
            torch.utils.data.SequentialSampler(
                torch.utils.data.Subset(
                    dataset,
                    torch.multinomial(
                        torch.ones(len(dataset)), num_samples, replacement=True
                    ),
                )
            )
            if sequential
            else torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=num_samples
            )
        ),
        batch_size=args.batchsize,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )


def _build_outdir_path(args) -> Path:
    name = ""
    if args.timestamp:
        timestamp = f'{datetime.now().strftime("%m-%d-%H-%M-%S")}'
        name = f"{timestamp}_"
    suffix = f"backbone_{args.backbone}"
    name = f"{name}{args.name}_{suffix}"
    if (Path(args.outdir) / name).exists():
        logger.info(f"Run name `{name}` already exists, prepending timestamp.")
        timestamp = f'{datetime.now().strftime("%m-%d-%H-%M-%S")}'
        name = f"{timestamp}_{name}"
    else:
        logger.info(f"Run name `{name}`")

    return Path(args.outdir) / name


def _convert_to_split_pairs(lst):
    """converts lst to a tuple of split pairs (ensuring backwards compatibility)

    [[0, .1, .2, .5]] -> [[0, .1], [.2, .5]]
    """

    # dont do anything if already converted
    if all(isinstance(x, (tuple, list)) and len(x) == 2 for x in lst):
        return tuple(lst)
    else:
        # flatten list
        lst = tuple(
            elem for x in lst for elem in (x if isinstance(x, (list, tuple)) else (x,))
        )
        if len(lst) % 2 == 0:
            return tuple(lst[i : i + 2] for i in range(0, len(lst), 2))
        else:
            raise ValueError(f"length of split {lst} should be even!")
        return


def _write_cams(data_visuals, model, device):
    for i, data in enumerate(data_visuals):
        vis = create_visuals(
            dataset=data,
            model=model,
            device=device,
            max_height=720,
            outdir=model.outdir / "visuals" / f"dataset_{i}",
        )


def main(args):
    if platform.system() == "Darwin":
        args.num_workers = 0
        logger.warning(
            "Setting num_workers to 0 to avoid MacOS multiprocessing issues."
        )

    if args.input_val is None:
        args.input_val = args.input_train

    args.split_train = _convert_to_split_pairs(args.split_train)
    args.split_val = _convert_to_split_pairs(args.split_val)

    outdir = _build_outdir_path(args)

    try:
        repo = git.Repo(Path(__file__).resolve().parents[1])
        args.tarrow_experiments_commit = str(repo.commit())
    except git.InvalidGitRepositoryError:
        pass

    tarrow.utils.seed(args.seed)

    device, n_gpus = tarrow.utils.set_device(args.gpu)
    if n_gpus > 1:
        raise NotImplementedError("Multi-GPU training not implemented yet.")

    augmenter = get_augmenter(args.augment)

    inputs = {}
    for inp, phase in zip((args.input_train, args.input_val), ("train", "val")):
        inputs[phase] = _get_paths_recursive(inp, args.read_recursion_level)
        logger.debug(f"{phase} datasets: {inputs[phase]}")

    logger.info("Build visualisation datasets.")
    data_visuals = tuple(
        _build_dataset(
            inp,
            split=(0, 1.0),
            size=None if args.cam_size is None else (args.cam_size,) * args.ndim,
            args=args,
            n_frames=args.frames,
            delta_frames=args.delta[-1:],
            permute=False,
            random_crop=False,
        )
        for inp in set([*inputs["train"], *inputs["val"]])
        # for inp in inputs["val"][-1:]
    )

    logger.info("Build training datasets.")
    data_train = ConcatDataset(
        _build_dataset(
            inp,
            split=split,
            size=(args.size,) * args.ndim,
            args=args,
            n_frames=args.frames,
            delta_frames=args.delta,
            augmenter=augmenter,
            reject_background=args.reject_background,
        )
        for split in args.split_train
        for inp in inputs["train"]
    )

    logger.info("Build validation datasets.")
    data_val = ConcatDataset(
        (
            _build_dataset(
                inp,
                split,
                size=(args.size,) * args.ndim,
                args=args,
                n_frames=args.frames,
                delta_frames=args.delta,
            )
            for split in args.split_val
            for inp in inputs["val"]
        )
    )

    loader_train = _create_loader(
        data_train,
        num_samples=args.train_samples_per_epoch,
        num_workers=args.num_workers,
        args=args,
    )

    loader_val = _create_loader(
        data_val,
        num_samples=args.val_samples_per_epoch,
        num_workers=0,
        args=args,
    )

    logger.info(f"Training set: {len(data_train)} images")
    logger.info(f"Validation set: {len(data_val)} images")

    model_kwargs = dict(
        backbone=args.backbone,
        projection_head=args.projhead,
        classification_head=args.classhead,
        n_frames=args.frames,
        n_input_channels=args.channels if args.channels > 0 else 1,
        n_features=args.features,
        device=device,
        symmetric=args.perm_equiv,
        outdir=outdir,
    )

    model = TimeArrowNet(**model_kwargs)
    model.to(device)

    logger.info(
        f"Number of params: {sum(p.numel() for p in model.parameters())/1.e6:.2f} M"
    )

    with open(outdir / "train_args.yaml", "tw") as f:
        yaml.dump(vars(args), f)

    assert args.ndim == 2

    model.fit(
        loader_train=loader_train,
        loader_val=loader_val,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_patience=args.lr_patience,
        epochs=args.epochs,
        steps_per_epoch=args.train_samples_per_epoch // args.batchsize,
        visual_datasets=tuple(
            Subset(d, list(range(0, len(d), 1 + (len(d) // args.cam_subsampling))))
            for d in data_visuals
        ),
        visual_dataset_frequency=args.visual_dataset_frequency,
        tensorboard=args.tensorboard > 0,
        save_checkpoint_every=args.save_checkpoint_every,
        lambda_decorrelation=args.decor_loss,
    )

    if args.write_final_cams:
        _write_cams(data_visuals, model, device)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
