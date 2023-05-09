import os
import logging
import re
import torch

logger = logging.getLogger(__name__)


def set_device(x):
    """Manages devices.

    Args:

        gpu (``int`` or ``str`` or list of ``int``):

            The IDs of the GPUs to use. Can be one of the following:
            - single integer
            - a comma-separated list of integers as a string
            - a list of integers
            - an interval `a-b` to use GPUs a, a+1, ..., b-1
            - `cpu`
            - `mps` to use Apple silicon GPU (metal performance shaders framework).
    """

    if isinstance(x, list):
        n_devices = len(x)
        val = ",".join(list(map(str, x)))
    elif isinstance(x, str):
        if re.search(r"^\d-\d$", x):
            n_devices = int(x[2]) - int(x[0]) + 1
            assert int(x[0]) < int(x[2])
            val = ",".join(list(map(str, list(range(int(x[0]), int(x[2]) + 1)))))
        elif re.search(r"^(\d,)*\d$", x):
            n_devices = len(x.split(","))
            val = x
        elif x == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS not available")
            device = torch.device(x)
            logger.info("PyTorch uses MPS.")
            return device, 1
        elif x == "cpu":
            device = torch.device(x)
            logger.info("PyTorch uses CPU.")
            return device, 1
        else:
            raise ValueError(f"{x=}")

    elif isinstance(x, int):
        n_devices = 1
        val = str(x)
    else:
        raise ValueError(f"{x=}")

    # Set before the first call to torch.cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = val

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"PyTorch uses GPUs {val}.")

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError(f"GPUs {x} not available.")

    # Set gpu number for gputools
    os.environ["gputools_id_platform"] = "0"
    os.environ["gputools_id_device"] = val[-1]

    return device, n_devices
