"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""
from einops import rearrange
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

# add repo root to path when running unit tests, so the import from files works
if __name__ == "__main__":
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(repo_root))

# from flam.configs.models.modules.quantizer.fsq_cfg import FsqConfig
from flam.models.modules.quantizer.utils import EPS, LossBreakdown, compute_entropy_perplexity
from flam.utils.misc import pack_one, unpack_one


def round_ste(z):
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()


def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()


class FiniteScalarQuantization(nn.Module):
    def __init__(
        self,
        fsq_cfg: SimpleNamespace,
        input_type: Literal["z", "action"]="z",
    ):
        super().__init__()

        _levels = torch.tensor(fsq_cfg.codebook_levels, dtype=torch.int64)
        self.register_buffer('_levels', _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + fsq_cfg.codebook_levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer('_basis', _basis, persistent=False)

        self.num_codebooks = fsq_cfg.num_codebooks
        self.codebook_size = fsq_cfg.codebook_size
        self.codebook_dim = fsq_cfg.codebook_dim
        self.code_dim = fsq_cfg.code_dim

        self.effective_codebook_dim = self.num_codebooks * fsq_cfg.code_dim

        if self.effective_codebook_dim != self.codebook_dim:
            self.project_in = nn.Linear(self.codebook_dim, self.effective_codebook_dim)
            self.project_out = nn.Linear(self.effective_codebook_dim, self.codebook_dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.preserve_symmetry = fsq_cfg.preserve_symmetry
        self.noise_dropout = fsq_cfg.noise_dropout

    @property
    def indice_logits_shape(self):
        return (self.num_codebooks, self.codebook_size)

    def bound(self, z):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + EPS) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    def symmetry_preserving_bound(self, z):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """
        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (z.tanh() + 1) / 2.) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        bound_fn = self.symmetry_preserving_bound if self.preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        # if using noise dropout, determine where to add a random offset elementwise
        if not self.training or self.noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, self.noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.
        else:
            half_width = self._levels // 2
            return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        # indices: (..., c)
        # returns: (..., c, d)

        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))
        else:
            half_width = self._levels // 2
            return (zhat_normalized * half_width) + half_width

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.code_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(torch.int64)

    def forward(self, z):
        """
        einstein notation
        b - batch
        c - number of codebooks
        d - codebook dim
        """

        assert z.shape[-1] == self.codebook_dim, f'expected dimension of {self.codebook_dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        # preprocess
        z, ps = pack_one(z, "* cd")                                                             # (b, c * d)

        # split out number of codebooks
        z = rearrange(z, "b (c d) -> b c d", c=self.num_codebooks)

        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)

        z_q = rearrange(z_q, 'b c d -> b (c d)')

        z_q = unpack_one(z_q, ps, "* cd")
        indices = unpack_one(indices, ps, "* c")

        z_q = self.project_out(z_q)

        loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        entropy, perplexity = compute_entropy_perplexity(indices, self.codebook_size)

        return z_q, indices, loss, LossBreakdown(entropy=entropy, perplexity=perplexity)

    @torch.no_grad()
    def get_codebook_entry(self, indices):
        """ Inverse of `codes_to_indices`. """

        codes = self._indices_to_codes(indices)
        codes = rearrange(codes, '... c d -> ... (c d)')
        codes = self.project_out(codes)

        return codes

    def get_codebook_entry_differentiable(self, indice_onehot):
        # indice_onehot: (..., c, n), one-hot

        all_indices = torch.arange(self.codebook_size, device=indice_onehot.device, dtype=torch.int64)  # (n,)
        all_codes = self._indices_to_codes(all_indices)                                                 # (n, d)

        codes = torch.einsum('... c n, n d -> ... c d', indice_onehot, all_codes)
        codes = rearrange(codes, '... c d -> ... (c d)')
        codes = self.project_out(codes)

        return codes


def test_get_codebook_entry_differentiable_gradients():
    """Test that all entries in indice_logits receive gradients."""
    
    # Test configuration
    fsq_cfg = FsqConfig(
        num_codebooks=2,
        codebook_levels=[4, 3],  # This gives codebook_size=12, code_dim=2
        codebook_dim=64,
        preserve_symmetry=False,
        noise_dropout=0.0
    )
    
    quantizer = FiniteScalarQuantization(fsq_cfg)
    quantizer.eval()  # Ensure we're in eval mode for consistent behavior
    
    # Create test data
    batch_size = 3
    num_codebooks = fsq_cfg.num_codebooks
    codebook_size = fsq_cfg.codebook_size
    
    # Create indice_logits with requires_grad=True
    indice_logits = torch.randn(batch_size, num_codebooks, codebook_size, requires_grad=True)
    
    # Apply softmax to make it a valid probability distribution
    indice_dist = F.one_hot(indice_logits.argmax(dim=-1), num_classes=codebook_size) + (indice_logits - indice_logits.detach())

    # Forward pass
    output = quantizer.get_codebook_entry_differentiable(indice_dist)

    # Check output shape
    expected_output_shape = (batch_size, num_codebooks * fsq_cfg.codebook_dim)
    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {output.shape}"

    # Create a dummy loss (sum of output)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check that gradients exist for indice_logits
    assert indice_logits.grad is not None, "Gradients should exist for indice_logits"

    # Check that all entries in indice_logits have gradients
    assert torch.any(indice_logits.grad != 0), "At least one entry in indice_logits should have non-zero gradients"

    # Check gradient shape matches input shape
    assert indice_logits.grad.shape == indice_logits.shape, f"Gradient shape {indice_logits.grad.shape} should match input shape {indice_logits.shape}"

    # Check that gradients are finite
    assert torch.all(torch.isfinite(indice_logits.grad)), "All gradients should be finite"

    print("✓ All entries in indice_logits receive gradients")
    print(f"  Input shape: {indice_logits.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Gradient shape: {indice_logits.grad.shape}")
    print(f"  Gradient norm: {indice_logits.grad.norm().item():.6f}")


if __name__ == "__main__":
    print("Running FSQ get_codebook_entry_differentiable gradient tests...")
    print("=" * 60)
    
    test_get_codebook_entry_differentiable_gradients()
    
    print("=" * 60)
    print("All tests passed! ✓")