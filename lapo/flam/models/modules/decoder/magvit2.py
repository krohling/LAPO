"""
modified from https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py
"""

from einops import rearrange

from typing import List

import torch
import torch.nn as nn

from flam.configs.models.modules.decoder.magvit2_cfg import Magvit2DecoderConfig
from flam.models.modules.common.activation import swish
from flam.models.modules.common.cnn_block import ResBlock
from flam.utils.misc import pack_one, unpack_one


class Magvit2Decoder(nn.Module):
    def __init__(self, decoder_cfg: Magvit2DecoderConfig, down_sizes: List[int], action_dim: int = 128):
        super().__init__()

        self.num_blocks = len(decoder_cfg.ch_mult)
        self.num_res_blocks = decoder_cfg.num_res_blocks

        block_in = decoder_cfg.ch * decoder_cfg.ch_mult[-1]
        self.conv_in = nn.Conv2d(
            decoder_cfg.z_channels + action_dim, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        down_sizes = list(reversed(down_sizes))
        ds_idx = 1  # skip the first one, which is the input channel size
        print(f"Magvit2Decoder: down_sizes: {down_sizes}, action_dim: {action_dim}")

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            in_filters = block_in + down_sizes[ds_idx] if res_idx == 0 else block_in
            self.mid_block.append(ResBlock(in_filters, block_in))

        ds_idx += 1
        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = decoder_cfg.ch * decoder_cfg.ch_mult[i_level]
            print(f"i_level: {i_level}, block_in: {block_in}, block_out: {block_out}, down_size: {down_sizes[ds_idx]}")
            print(f"AdaptiveGroupNorm(decoder_cfg.z_channels + action_dim, block_in): {decoder_cfg.z_channels + action_dim}, {block_in}")
            self.adaptive.insert(0, AdaptiveGroupNorm(decoder_cfg.z_channels + action_dim, block_in))
            for i_block in range(self.num_res_blocks):
                in_filters = block_in + down_sizes[ds_idx] if i_block == 0 else block_in
                block.append(ResBlock(in_filters, block_out))
                block_in = block_out

            ds_idx += 1
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = UpSampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, 3, kernel_size=(3, 3), padding=1)

        print(f"final ds_idx: {ds_idx}, len(down_sizes): {len(down_sizes)}")

    def forward(self, z, enc_features: List, action: torch.Tensor=None):
        # :arg z:  (..., H_feat, W_feat, D)
        # :return: (..., 3, H, W)

        enc_features = list(reversed(enc_features))
        feat_idx = 0

        # preprocess
        z, ps = pack_one(z, "* h w d")                      # (..., H, W, D) -> (B, H, W, D)
        z = rearrange(z, "b h w d -> b d h w")
        print(f"z.shape after rearrange: {z.shape}")

        if action is not None:
            _, _, h, w = z.shape
            enc_features[feat_idx] = action[:, :, None, None].repeat(1, 1, h, w)

        print(f"features[feat_idx].shape: {enc_features[feat_idx].shape}")
        z = torch.cat([z, enc_features[feat_idx]], dim=1)
        feat_idx += 1
        
        style = z.clone()  # for adaptive groupnorm

        print(f"z.shape before conv_in: {z.shape}")
        z = self.conv_in(z)
        print(f"z.shape after conv_in: {z.shape}")
        z = torch.cat([z, enc_features[feat_idx]], dim=1)
        print(f"z.shape after first feat cat: {z.shape}")
        feat_idx += 1

        # mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        print(f"z.shape after mid: {z.shape}")
        # z = torch.cat([z, features[feat_idx]], dim=1)
        # print(f"z.shape after mid feat cat: {z.shape}")
        # feat_idx += 1

        # upsample
        for i_level in reversed(range(self.num_blocks)):
            # pass in each resblock first adaGN
            print(f"i_level: {i_level}, z.shape before up: {z.shape}")
            z = self.adaptive[i_level](z, style)


            z = torch.cat([z, enc_features[feat_idx]], dim=1)
            feat_idx += 1
            for i_block in range(self.num_res_blocks):
                print(f"i_block: {i_block}, z.shape before up block: {z.shape}")
                z = self.up[i_level].block[i_block](z)

            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        # postprocess
        z = unpack_one(z, ps, "* d h w")                    # (B, 3, H, W) -> (..., 3, H, W)

        print(f"final feat_idx: {feat_idx}, len(features): {len(enc_features)}")

        return z


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size ** 2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


class UpSampler(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, z):
        print(f"AdaptiveGroupNorm: x.shape: {x.shape}, z.shape: {z.shape}")
        B, C, _, _ = x.shape

        # calculate var for scale
        scale = rearrange(z, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps  # not unbias
        scale = scale.sqrt()
        print(f"scale.shape: {scale.shape}")
        scale = self.gamma(scale).view(B, C, 1, 1)

        # calculate mean for bias
        bias = rearrange(z, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x
