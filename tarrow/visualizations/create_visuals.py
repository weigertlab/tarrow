from types import SimpleNamespace
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import imageio
from scipy import ndimage as ndi
from matplotlib import cm
import torch

# from ..utils import normalize
from tarrow.utils import normalize, str2bool


def create_visuals(
    dataset,
    model,
    device: str,
    max_height: int = np.inf,
    alpha_cam: float = 0.7,
    return_feats: bool = False,
    norm_cam=True,
    outdir: str = None,
    fps: int = 5,
    file_format: str = "tiff",
):
    res = defaultdict(list)

    for i, (x, _) in tqdm(enumerate(dataset), total=len(dataset), desc="creating cams"):
        zoom_factor = (
            max_height / x.shape[-2]
            if max_height > 0 and x.shape[-2] > max_height
            else 1
        )

        cam = model.gradcam(
            x,
            class_id=0,
            norm=False,
            # tile_size=(1280, 1280),
            tile_size=None,
        )

        if return_feats:
            feat = (
                model(x.unsqueeze(0).to(model.device), mode="projection")[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            if zoom_factor != 1:
                feat = ndi.zoom(feat, (1,) + (zoom_factor,) * 2, order=1)
            res["feats"].append(feat)

        raw = x[:, 0].detach().cpu()
        if zoom_factor != 1:
            raw = ndi.zoom(raw, (1,) + (zoom_factor,) * 2, order=1)
            cam = ndi.zoom(cam, (zoom_factor,) * 2, order=1)

        res["raws"].append(raw)
        res["cam"].append(cam)

    torch.cuda.empty_cache()
    for k, v in res.items():
        res[k] = np.stack(v)

    res = SimpleNamespace(**res)
    res.raws = normalize(res.raws)

    if norm_cam:
        res.cam = normalize(res.cam, 0.1, 99.99)

    res.raw_with_time = res.raws
    res.raws = res.raws[:, 0]

    def _to_uint8(x):
        return (255 * np.clip(x, 0, 1)).astype(np.uint8)

    # Write to file
    if outdir is not None:
        res_rgb = dict()
        res_rgb["raws"] = res.raws
        res_rgb["cam"] = res.cam

        # res_rgb["overlays"] = (1 - alpha_cam) * res_rgb["raws"] + alpha_cam * res_rgb[
        #     "cam"
        # ]

        # for k, v in res_rgb.items():
        # res_rgb[k] = _to_uint8(v)

        res_rgb = SimpleNamespace(**res_rgb)

        outdir = Path(outdir).expanduser()
        for name, visual in vars(res_rgb).items():
            if visual.ndim > 4:
                continue
            subdir = outdir / name
            subdir.mkdir(parents=True, exist_ok=True)
            for i, x in tqdm(enumerate(visual), leave=False, desc=f"Write {name}"):
                imageio.imsave(
                    subdir / f"{name}_{i:05d}.{file_format}",
                    x,
                    compression="zlib",
                )

    return res


def get_argparser():
    parser = argparse.ArgumentParser(
        description="tarrow-create-visuals",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        nargs="+",
        help="input image folder or files",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="model checkpoint file (.pt)"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True, help="output folder that stores cams"
    )
    parser.add_argument("--channels", type=int, default=0)
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="use updated model instead of torch.load",
    )
    parser.add_argument("-n", "--n_images", type=int, default=None)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_height", type=int, default=2048, help="Limits size of output images"
    )
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--divby", type=int, default=1)
    parser.add_argument("--file_format", default="tiff", choices=["jpg", "png", "tiff"])
    parser.add_argument(
        "--size",
        default=None,
        type=int,
        nargs="+",
        help="Limit size of input image to the model",
    )
    parser.add_argument("--norm_cam", type=str2bool, default=False)
    return parser


def write_visuals(args=None):
    from tarrow.data import TarrowDataset
    from tarrow.models import TimeArrowNet

    if args is None:
        parse = get_argparser()
        args = parse.parse_args()

    print(args)

    model = TimeArrowNet.from_folder(
        Path(args.model),
        map_location=args.device,
        from_state_dict=args.update,
        ignore_commit=True,
    )

    model.eval()

    if len(args.input) == 1:
        fname = args.input[0]
    else:
        fname = args.input

    dataset = TarrowDataset(
        imgs=fname,
        n_frames=args.frames,
        n_images=args.n_images,
        delta_frames=[args.delta],
        subsample=args.subsample,
        channels=args.channels,
        size=args.size,
        permute=False,
        random_crop=False,
        device=args.device,
    )

    res = create_visuals(
        dataset,
        model,
        device=args.device,
        max_height=args.max_height,
        return_feats=False,
        alpha_cam=args.alpha,
        outdir=args.outdir,
        norm_cam=args.norm_cam,
        fps=args.fps,
        file_format=args.file_format,
    )


if __name__ == "__main__":
    write_visuals()
