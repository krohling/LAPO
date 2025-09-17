from pathlib import Path
from einops import pack, unpack

import torch
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