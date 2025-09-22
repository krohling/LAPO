from pathlib import Path
from einops import pack, unpack

import dataclasses

from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

REPO_PATH = repo_path = Path(__file__).resolve().parents[1]

def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


@torch.no_grad()
def compute_entropy_perplexity(
    indices: torch.Tensor,
    codebook_size: int,
):
    """
    Compute entropy of the indices.

    Args:
        indices: (..., c)
            - c: number of codebooks
        codebook_size: int

    Returns:
    """
    EPS = 1e-5
    indices, _ = pack_one(indices, "* c")                               # (b, c)
    indices_onehot = F.one_hot(indices, codebook_size)                  # (b, c, n)

    avg_probs = indices_onehot.float().mean(dim=0)                      # (c, n)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPS))
    perplexity = torch.exp(entropy)

    return entropy, perplexity


def check_configs_match(
    dict1,
    dict2,
    prefix: str = "",
    keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    only_check_common_keys: bool = False,
):
    if keys is None:
        keys = []
    if exclude_keys is None:
        exclude_keys = []

    # Compare dict keys
    if not keys:
        if only_check_common_keys:
            keys = set(dict1.keys()) & set(dict2.keys())
        else:
            keys = set(dict1.keys()) | set(dict2.keys())
    else:
        assert not exclude_keys, "exclude_keys is not supported when keys is specified."
        assert not only_check_common_keys, "only_check_common_keys is not supported when keys is specified."

    for key in keys:
        if key in exclude_keys:
            continue

        prefix_key = f"{prefix}.{key}" if prefix else key

        if key not in dict1:
            raise ValueError(f"Key {prefix_key} not found in dict1.")

        if key not in dict2:
            raise ValueError(f"Key {prefix_key} not found in dict2.")

    # Compare dict values
    for key in keys:
        if key in exclude_keys:
            continue

        value1, value2 = dict1[key], dict2[key]

        prefix_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value1, torch.Tensor):
            assert isinstance(value2, torch.Tensor)
            value1, value2 = value1.cpu(), value2.cpu()
            if not (value1.shape == value2.shape and torch.allclose(value1, value2)):
                raise ValueError(f"Mismatch found at {prefix_key}.")
        elif isinstance(value1, dict):
            assert isinstance(value2, dict)
            check_configs_match(
                value1,
                value2,
                prefix=prefix_key,
                only_check_common_keys=only_check_common_keys,
            )
        else:
            if (np.isscalar(value1) and value1 != value2) or (not np.isscalar(value1) and np.any(value1 != value2)):
                raise ValueError(f"Mismatch found at {prefix_key}: {value1} != {value2}")

def get_checkpoint_config(checkpoint_path: Path, cfg_key: str) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_PATH / checkpoint_path
    assert checkpoint_path.exists(), f"=> no checkpoint found at '{checkpoint_path}'"

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint[cfg_key] if cfg_key in checkpoint else {}

def load_checkpoint(
    module: nn.Module,
    checkpoint_path: Union[Path, None],
    cfg_key: str,
    cfg_exclude_keys: Optional[Tuple] = (),
    only_check_common_keys: bool = False,
    strict: bool = True,
):
    if checkpoint_path is None:
        return

    module_name = module.__class__.__name__

    # check if checkpoint_path is valid
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_PATH / checkpoint_path
    assert checkpoint_path.exists(), f"=> no {module_name} checkpoint found at '{checkpoint_path}'"

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # verify config is the same
    check_configs_match(
        # dataclasses.asdict(getattr(module, cfg_key)),
        getattr(module, cfg_key),
        checkpoint[cfg_key],
        exclude_keys=cfg_exclude_keys,
        only_check_common_keys=only_check_common_keys,
    )

    # load model state dict
    model_state_dict = checkpoint["model"]

    # remove ddp prefix if present
    model_state_dict = {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in model_state_dict.items()
    }

    # load model state dict and print message
    msg = module.load_state_dict(model_state_dict, strict=strict)
    print(f"=> {module_name} loaded from checkpoint: {checkpoint_path} with msg:")

    if not msg.missing_keys and not msg.unexpected_keys:
        print("<All keys matched successfully>.")

    if msg.missing_keys:
        print("Missing keys:\n\t" + "\n\t".join(msg.missing_keys) + "\n")

    if msg.unexpected_keys:
        print("Unexpected keys:\n\t" + "\n\t".join(msg.unexpected_keys) + "\n")