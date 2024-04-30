from collections import defaultdict
import logging
from pathlib import Path
from typing import Sequence, Union
import json
from time import time as now

import dill
import numpy as np
from scipy.ndimage import zoom
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision.utils import make_grid

import yaml
from .backbones import get_backbone
from .proj_heads import ProjectionHead
from .class_heads import ClassificationHead
from .losses import DecorrelationLoss
from ..utils import normalize, tile_iterator
from ..visualizations import create_visuals
from ..visualizations import cam_insets

logger = logging.getLogger(__name__)


class NoOutputFolderException(Exception):
    def __init__(self) -> None:
        super().__init__("Model doesnt have an associated output folder!")


def _tensor_random_choice(
    x: torch.Tensor, n_samples: Union[int, float]
) -> torch.Tensor:
    """randomly select n_samples with replacement from a flat tensor
    if n_samples is float, will be interpreted as fraction
    """
    assert x.ndim == 1

    if isinstance(n_samples, float):
        assert 0 <= n_samples <= 1.0
        n_samples = int(len(x) * n_samples)

    n_samples = min(max(1, n_samples), len(x))
    idx = np.random.randint(0, len(x), n_samples)
    return x[idx]


def _git_commit():
    """returns the git commit hash of the current repository if it exists, otherwise None (for debugging purposes)"""
    import git

    try:
        return str(git.Repo(Path(__file__).resolve().parents[2]).commit())
    except git.exc.InvalidGitRepositoryError:
        return None


class TimeArrowNet(nn.Module):
    def __init__(
        self,
        backbone="unet",
        projection_head="minimal_batchnorm",
        classification_head="minimal",
        n_frames=2,
        n_features=16,
        n_input_channels=1,
        device="cpu",
        symmetric=False,
        outdir: str = None,
        commit=None,
    ):
        """Full TAP model consisting of single-image backbone, single_image projection head and joint classification head.

        Args:
            backbone:
                Dense network architecture.
            projection_head:
                Dense projection head architecture.
            classification_head:
                Classification head architecture.
            n_frames:
                Number of input frames.
            n_features:
                Number of output features from the backbone.
            n_input_channels:
                Number of input channels.
            device:
                Device to run the model on.
            symmetric:
                If `True`, use permutation-equivariant classification head.
            outdir:
                Output directory for model checkpoints and tensorboard logs.
            commit:
                Commit hash of the current git commit, used for model loading.
        """

        super().__init__()

        model_kwargs = dict(
            backbone=backbone,
            projection_head=projection_head,
            classification_head=classification_head,
            n_frames=n_frames,
            n_features=n_features,
            n_input_channels=n_input_channels,
            symmetric=symmetric,
            outdir=str(outdir),
            commit=commit,
        )

        self.n_features = n_features
        self.backbone, self.bb_n_feat = get_backbone(backbone, n_input=n_input_channels)

        self.projection_head = ProjectionHead(
            in_features=self.bb_n_feat,
            out_features=n_features,
            mode=projection_head,
        )

        self.classification_head = ClassificationHead(
            in_features=n_features,
            n_frames=n_frames,
            out_features=n_features,
            n_classes=n_frames,
            mode=classification_head,
            symmetric=symmetric,
        )

        self.n_frames = n_frames
        self.device = device

        # Hook for cam layer
        self.proj_activations = None
        self.proj_gradients = None
        self.projection_head.register_forward_hook(self.get_activation)

        model_kwargs["commit"] = _git_commit()

        self.outdir = outdir
        if self.outdir is not None:
            with open(self.outdir / "model_kwargs.yaml", "tw") as f:
                yaml.dump(model_kwargs, f)

    @property
    def outdir(self):
        return self._outdir

    @outdir.setter
    def outdir(self, path):
        if path is None:
            self._outdir = None
            return

        path = Path(path)
        if path.exists():
            raise RuntimeError(f"Model folder already exists {path}.")
        self._outdir = path
        for sub in (".", "tb", "visuals"):
            (self._outdir / sub).mkdir(exist_ok=False, parents=True)

    def get_activation(self, model, input, output):
        self.proj_activations = output.detach()

    def get_gradients(self, grad):
        self.proj_gradients = grad.detach()

    def forward(self, x, mode="classification"):
        """
        Args:
            x: Tensor of size B, T, C, H, W
            mode: str, can be
                'classification'
                'projection'
                'both'

        Returns:
            final: Tensor of size B, n_classes
            (projections: Tensor of size B, T, n_features, H, W)

        """

        # Flatten timepoints into the batch dimension
        s_in = x.shape
        x = x.flatten(end_dim=1)

        # backbone features for every timepoint independently
        x = self.backbone(x)
        s_out = x.shape

        # -> (Batch, Timepoints, n_backbone_features, D0, ..., Dn)
        features = x.reshape(s_in[:2] + (self.bb_n_feat,) + s_out[2:])
        # for id backbone x = x.reshape(s[:2] + (1,) + x.shape[2:])

        # projected features for every timepoint independently
        # -> (Batch, Timepoints, n_features, D0, ..., Dn)
        projections = self.projection_head(features)

        # Store gradients for gradcam
        if projections.requires_grad:
            projections.register_hook(self.get_gradients)

        # classification merges features
        if mode == "classification":
            final = self.classification_head(projections)
            return final
        elif mode == "projection":
            return projections
        elif mode == "both":
            final = self.classification_head(projections)
            return final, projections
        else:
            raise ValueError(f"unknown mode {mode}")

    def gradcam(
        self, x, class_id=0, norm=False, all_frames=False, tile_size=(512, 512)
    ):
        """Grad-CAM with respect to projection layer

        Args:
            x:
                Tensor of size T, C, H, W.
            class_id:
                Ground-truth class id. 0 when frame order is not altered.
            norm:
                If `True`, normalize output to (0,1).
            all_frames:
                If `True`, sum over all frames. If `False`, only consider first
                time step that is put into classification head.
            tile_size:
                tuple of ints, size of tiles to process

        Returns:
            gradcam: Tensor of size H, W.
        """

        if is_training := self.training:
            self.eval()

        assert x.ndim == 4, f"{x.ndim=}"

        def _get_alpha_and_activation(_x: torch.Tensor):
            self.zero_grad()
            u = self(_x.unsqueeze(0))[0]
            u = u[class_id]
            u.backward()
            A = self.proj_activations[0].detach()
            alpha = self.proj_gradients[0].detach()

            return alpha, A

        if tile_size is None or torch.all(
            torch.as_tensor(tile_size) >= torch.as_tensor(x.shape[-2:])
        ):
            x = torch.as_tensor(x, device=self.device)
            alpha, A = _get_alpha_and_activation(x)
        else:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            shape = (x.shape[0], self.n_features) + x.shape[2:]
            alpha = torch.zeros(shape, device=self.device)
            A = torch.zeros(shape, device=self.device)
            # tile input to reduce memory requirements
            blocksize = x.shape[:2] + tuple(
                min(t, s) for t, s in zip(tile_size, x.shape[2:])
            )
            tq = tile_iterator(
                x,
                blocksize=blocksize,
                padsize=(0, 0, min(64, tile_size[0] // 4), min(64, tile_size[1] // 4)),
                mode="reflect",
            )
            # for tile, s_src, s_dest in tqdm(tq, desc="grad cam with tiling"):
            for tile, s_src, s_dest in tq:
                tile = torch.as_tensor(tile, device=self.device)
                _alpha, _A = _get_alpha_and_activation(tile)
                if _alpha.shape[-2:] != tile.shape[-2:]:
                    raise NotImplementedError(
                        "Tiled CAMs only for nets with input size == output size"
                    )
                s_src = (slice(None),) * 2 + s_src[2:]
                s_dest = (slice(None),) * 2 + s_dest[2:]
                alpha[s_src] = _alpha[s_dest]
                A[s_src] = _A[s_dest]

        # get the correct normalization (as A is already average summed)
        alpha = torch.sum(alpha, (-1, -2))
        cam = torch.einsum("tc,tcyx->tyx", alpha, A)

        cam = torch.abs(cam)

        if all_frames:
            cam = cam.sum(0)
        else:
            cam = cam[0]

        if norm:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cam.cpu().numpy()
        factor = np.array(x.shape[-2:]) / np.array(cam.shape)
        # upsample CAM to input image size with interpolation
        if not np.all(factor - 1 == 0):
            cam = zoom(cam, factor, order=1)

        if is_training:
            self.train()

        return cam

    def embedding(self, x, layer=0):
        """Returns the feature maps from the n-th last layer.

        Args:
            x:
                Input of shape B, T, C, H, W.
            layer:
                Features are taken from that projection head layer.
                Layers are numbered from last to first, starting at 0.
        """
        _ = self(x, mode="projection")
        n = len(self.projection_head.features)
        if n <= layer:
            raise ValueError(
                f"{n} available feature layers. Embedding for layer {layer} not available."
            )

        features = self.projection_head.features[layer]
        features = features.reshape(x.shape[:2] + features.shape[1:])
        return features

    def save(self, prefix="model", which="both", exist_ok: bool = False, outdir=None):
        """Save entire model.

        Uses `dill` as pickling module to be able to pickle the hooks(closures).
        Refer to https://github.com/pytorch/pytorch/issues/1148.

        Args:
            prefix: File name prefix.
            which: can be 'full', 'state', or 'both
            exist_ok: If `True`, overwrite existing files.
            outdir: Output directory. If `None`, use `self.outdir`.
        """
        if outdir is None:
            outdir = self.outdir

        outdir.mkdir(exist_ok=True, parents=True)

        if which == "both" or which == "full":
            fpath = outdir / f"{prefix}_full.pt"
            if not exist_ok and fpath.exists():
                raise FileExistsError(fpath)
            torch.save(self, fpath, pickle_module=dill)

        if which == "both" or which == "state":
            fpath = outdir / f"{prefix}_state.pt"
            if not exist_ok and fpath.exists():
                raise FileExistsError(fpath)
            torch.save(self.state_dict(), fpath)

    @classmethod
    def from_folder(
        cls,
        model_folder,
        from_state_dict=False,
        map_location="cpu",
        ignore_commit=True,
        state_dict_path="model_state.pt",
    ):
        """Load model from folder, either from state dict or from full model.

        Args:
            model_folder: Path to folder containing model.
            from_state_dict: If `True`, load from state_dict, otherwise from `model_full.pt`.
            map_location: Device where the params of the model will be loaded to.
            ignore_commit:
                If `True`, ignore mismatched commits of the saved model and
                current code, and try to load the model anyway.
            state_dict_path: Name of the state dict file.

        Returns:
            model: TimeArrowModel instance.
        """
        model_folder = Path(model_folder)
        logging.info(f"Loading model from {model_folder}")
        kwargs = yaml.safe_load(open(model_folder / "model_kwargs.yaml", "rt"))

        if not ignore_commit:
            _commit = _git_commit()
            if "commit" in kwargs and kwargs["commit"] != _commit:
                raise RuntimeError(
                    f"Git commit of saved model ({kwargs['commit']}) does not match current commit of tarrow repo ({repo.commit()}). Set `ignore_commit` parameter to `True` to proceed."
                )

        if "commit" in kwargs:
            del kwargs["commit"]

        if from_state_dict:
            kwargs["device"] = map_location
            state_dict = torch.load(
                model_folder / state_dict_path, map_location=map_location
            )
            logging.info(f"Loading state dict {state_dict_path}")
            kwargs["outdir"] = None
            model = TimeArrowNet(**kwargs)
            model.load_state_dict(state_dict)
            model.to(map_location)
        else:
            model = torch.load(
                model_folder / "model_full.pt", map_location=map_location
            )
            model.device = torch.device(map_location)
            model.outdir = None

        return model

    def save_example_images(self, loader, tb_writer, phase):
        """Write example images to tensorboard.

        Args:
            loader: Torch Dataloader.
            tb_writer: Tensorboard writer.
            phase: 'train' or 'val'.
        """
        logger.info("Write example images to tensorboard.")
        example_imgs, _ = next(iter(loader))
        logger.debug(f"{example_imgs.shape=}")

        def write_to_tb(imgs, name):
            # only first channel
            imgs = imgs[:, :, 0, ...]
            # imgs = (imgs - imgs.min(dim=(1,2,3), keepdim=True)[0]) / (
            #     imgs.max(dim=(1,2,3), keepdim=True)[0] - imgs.min(dim=(1,2,3), keepdim=True)[0]
            # )
            # logger.debug(
            # f"{name} min = {imgs.min():.2f} max = {imgs.max():.2f} mean {imgs.mean():.2f}"
            # )
            for i in range(imgs.shape[1]):
                tb_writer.add_image(
                    name,
                    make_grid(
                        imgs[:64, i : i + 1, ...],
                        scale_each=True,
                        value_range=(0, 1),
                        padding=0,
                    ),
                    global_step=i,
                )

        write_to_tb(example_imgs, f"example_images/{phase}")

    def fit(
        self,
        loader_train,
        loader_val,
        lr,
        lr_patience,
        epochs,
        steps_per_epoch,
        lr_scheduler="plateau",
        visual_datasets=(),
        visual_dataset_frequency=10,
        tensorboard=True,
        save_checkpoint_every=100,
        weight_decay=1e-6,
        lambda_decorrelation=0.01,
    ):
        """Train model.

        Args:
            loader_train:
                Torch Dataloader for training.
            loader_val:
                Torch Dataloader for validation.
            lr:
                Learning rate.
            lr_patience:
                Patience for learning rate scheduler.
            epochs:
                Number of training epochs.
            steps_per_epoch:
                Number of training steps (samples / batch size) per epoch.
            lr_scheduler:
                Learning rate scheduler. Either 'plateau' or 'cyclic'.
            visual_datasets:
                Sequence of datasets to visualize in tensorboard.
            visual_dataset_frequency (int):
                Save attribution maps to tensorboard every n epochs.
            tensorboard:
                If `True`, write tensorboard logs.
            save_checkpoint_every (int):
                Save model checkpoint every n epochs.
            weight_decay:
                Weight decay for optimizer.
            lambda_decorrelation:
                Prefactor of decorrelation loss.
        """
        assert self.outdir is not None

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        if lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, patience=lr_patience, verbose=True
            )
        elif lr_scheduler == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr,
                max_lr=lr * 10,
                cycle_momentum=False,
                step_size_up=steps_per_epoch,
                scale_mode="cycle",
                scale_fn=lambda x: 0.9**x,
                verbose=False,
            )
        else:
            raise ValueError()

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        criterion_decorr = DecorrelationLoss()

        # save visuals (e.g. cams) in model folder and tensorboard
        def _save_visuals(
            dataset: Sequence[Dataset], tb_writer, epoch: int, save_features=False
        ):
            n_insets = 8
            inset_size = 48

            for i, data in enumerate(dataset):
                vis = create_visuals(
                    dataset=data,
                    model=self,
                    device=self.device,
                    max_height=480,
                    outdir=None,
                    return_feats=save_features,
                )
                if tb_writer is not None:
                    for j, (raw_with_time, cam) in tqdm(
                        enumerate(zip(vis.raw_with_time, vis.cam)),
                        desc="Write insets",
                        total=len(vis.raw_with_time),
                        leave=True,
                    ):
                        fig, _, _ = cam_insets(
                            xs=raw_with_time,
                            cam=cam,
                            n_insets=n_insets,
                            w_inset=inset_size,
                            main_frame=0,
                        )
                        tb_writer["cams"].add_figure(
                            f"dataset_{i}/{j}", fig, global_step=epoch
                        )

                    if save_features:
                        for j, feats in tqdm(
                            enumerate(vis.feats),
                            desc="Write features",
                            total=len(vis.feats),
                            leave=True,
                        ):
                            for k, feat in enumerate(feats):
                                tb_writer["features"].add_image(
                                    f"features_{i}/{j}",
                                    normalize(feat[None], clip=True),
                                    global_step=k,
                                )

        def _model_step(loader, phase="train", title="Training"):
            start = now()
            if phase == "train":
                self.train()
            else:
                self.eval()

            losses, losses_decorr, accs = 0.0, 0.0, 0.0
            count, sum_preds = 0, 0

            with torch.set_grad_enabled(phase == "train"):
                pbar = tqdm(loader, leave=False)

                for x, y in pbar:
                    x, y = x.to(self.device), y.to(self.device)

                    if phase == "train":
                        optimizer.zero_grad()

                    out, pro = self(x, mode="both")

                    if out.ndim > 2:
                        y = torch.broadcast_to(
                            y.unsqueeze(1).unsqueeze(1), (y.shape[0],) + out.shape[-2:]
                        )
                        loss = criterion(out, y)
                        loss = torch.mean(loss, tuple(range(1, loss.ndim)))
                        y = y[:, 0, 0]
                        u_avg = torch.mean(out, tuple(range(2, out.ndim)))

                    else:
                        u_avg = out
                        loss = criterion(out, y)

                    pred = torch.argmax(u_avg.detach(), 1)

                    loss = torch.mean(loss)

                    # decorrelation loss
                    pro_batched = pro.flatten(0, 1)
                    loss_decorr = criterion_decorr(pro_batched)

                    loss_all = loss + lambda_decorrelation * loss_decorr
                    if phase == "train":
                        loss_all.backward()
                        optimizer.step()
                        if lr_scheduler == "cyclic":
                            scheduler.step()

                    sum_preds += pred.sum().item()

                    count += pred.shape[0]
                    acc = torch.mean((pred == y).float())
                    losses += loss.item() * pred.shape[0]
                    losses_decorr += loss_decorr.item() * pred.shape[0]
                    accs += acc.item() * pred.shape[0]
                    pbar.set_description(
                        f"{losses/count:.6f} | {losses_decorr/count:.6f} ({phase})"
                    )

            metrics = dict(
                loss=losses / count,
                loss_decorr=losses_decorr / count,
                accuracy=accs / count,
                pred1_ratio=sum_preds / count,
                lr=scheduler.optimizer.param_groups[0]["lr"],
            )
            logger.info(
                f"{title} ({int(now() - start):3}s) Loss: {metrics['loss']:.5f} Decorr: {metrics['loss_decorr']:.5f} ACC: {metrics['accuracy']:.5f}"
            )
            return metrics

        if tensorboard:
            tb_writer = dict(
                (key, SummaryWriter(str(self.outdir / "tb" / key)))
                for key in ("train", "val", "cams", "features")
            )
        else:
            tb_writer = None

        if tb_writer is not None and visual_dataset_frequency > 0:
            self.save_example_images(loader_train, tb_writer["train"], "train")
            self.save_example_images(loader_val, tb_writer["val"], "val")

        metrics = defaultdict(lambda: defaultdict(list))

        for i in range(1, epochs + 1):
            metrics_train = _model_step(
                loader_train,
                "train",
                f"--- Training   ({i}/{epochs})",
            )
            metrics_val = _model_step(
                loader_val, "val", f"+++ Validation ({i}/{epochs})"
            )

            if lr_scheduler == "plateau":
                scheduler.step(metrics_val["loss"])

            for loader, phase, metr in zip(
                (loader_train, loader_val),
                ("train", "val"),
                (metrics_train, metrics_val),
            ):
                for k, v in metr.items():
                    metrics[phase][k].append(v)
                metrics[phase]["steps"].append((i + 1) * len(loader))

            if self.outdir is not None:
                if visual_dataset_frequency > 0 and i % visual_dataset_frequency == 0:
                    _save_visuals(visual_datasets, tb_writer, epoch=i)

            with open(self.outdir / "losses.json", "wt") as f:
                json.dump(metrics, f)

            if tb_writer is not None:
                for phase, met in metrics.items():
                    for k, v in met.items():
                        tb_writer[phase].add_scalar(k, v[-1], i)

                for tbw in tb_writer.values():
                    tbw.flush()

            if i % save_checkpoint_every == 0:
                logger.info(f"Saving checkpoint: epoch = {i}")
                self.save(
                    prefix=f"epoch_{i:05d}",
                    which="state",
                    outdir=self.outdir / "checkpoints",
                )

            target = metrics["val"]["loss"]
            if np.argmin(target) + 1 == i:
                logger.info(f"Saving best model: epoch = {i} val_loss = {target[-1]}")
                self.save(which="both", exist_ok=True)

        if tb_writer is not None:
            _save_visuals(visual_datasets, tb_writer, epoch=i, save_features=True)

        if tb_writer is not None:
            for tbw in tb_writer.values():
                tbw.close()
