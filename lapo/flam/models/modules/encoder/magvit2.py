"""
modified from https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py
"""

from typing import Union, Tuple
from einops import rearrange

import torch
import torch.nn as nn

from flam.configs.models.modules.encoder.magvit2_cfg import Magvit2EncoderConfig
from flam.models.modules.common.activation import swish
from flam.models.modules.common.cnn_block import ResBlock
from flam.utils.misc import pack_one, unpack_one

def merge_TC_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B, T*C, H, W)"""
    return x.view(x.shape[0], -1, *x.shape[3:])

class Magvit2Encoder(nn.Module):
    def __init__(self, 
                    encoder_cfg: Magvit2EncoderConfig,
                    image_size: Union[int, Tuple[int, int]], 
                    sub_traj_len: int, 
                    action_dim: int=None
                ):
        super().__init__()
        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size

        self.num_res_blocks = encoder_cfg.num_res_blocks
        self.num_blocks = len(encoder_cfg.ch_mult)

        self.down_sizes = []
        ch_in = 3 * sub_traj_len + (action_dim if action_dim is not None else 0)
        self.down_sizes.append(ch_in)

        self.conv_in = nn.Conv2d(ch_in, encoder_cfg.ch, kernel_size=(3, 3), padding=1, bias=False)
        self.down_sizes.append(encoder_cfg.ch)

        # construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(encoder_cfg.ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = encoder_cfg.ch * in_ch_mult[i_level]
            block_out = encoder_cfg.ch * encoder_cfg.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)

            self.down.append(down)
            self.down_sizes.append(block_out)

        # mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        self.down_sizes.append(block_in)

        # end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, encoder_cfg.z_channels, kernel_size=(1, 1))

        # calculate feature shape
        down_sample_scale = 2 ** (self.num_blocks - 1)
        assert H % down_sample_scale == 0 and W % down_sample_scale == 0
        self.H_feat = H // down_sample_scale
        self.W_feat = W // down_sample_scale
        self.feat_dim = encoder_cfg.z_channels

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def H(self):
        return self.H_feat

    @property
    def W(self):
        return self.W_feat

    @property
    def N(self):
        return self.H * self.W

    @property
    def D(self):
        return self.feat_dim

    def forward(self, x, action=None):
        # :arg x:  (..., 3, H, W), normalized to [0, 1]
        # :return: (..., H_feat, W_feat, D)

        # print("*****Magvit2Encoder.forward*****")

        # preprocess
        # x, ps = pack_one(x, "* d h w")                      # (..., 3, H, W) -> (B, 3, H, W)
        x = merge_TC_dims(x)  # (B, T, 3, H, W) -> (B, T*3, H, W)

        if action is not None:
            _, _, h, w = x.shape
            action = action[:, :, None, None]
            x = torch.cat([x, action.repeat(1, 1, h, w)], dim=1)

        # down
        feat = []
        # print(f"conv_in: {x.shape}")
        x = self.conv_in(x)
        feat.append(x)

        for i_level in range(self.num_blocks):
            # print(f"i_level {i_level}: {x.shape}")
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
            
            feat.append(x)

        # mid
        # print(f"mid: {x.shape}")
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        feat.append(x)

        # print(f"norm_out & conv_out: {x.shape}")
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        # postprocess
        x = rearrange(x, "b d h w -> b h w d")
        # x = unpack_one(x, ps, "* h w d")                    # (B, H_feat, W_feat, D) -> (..., H_feat, W_feat, D)

        assert x.shape[-3:] == (self.H_feat, self.W_feat, self.feat_dim)

        # print(f"Magvit2Encoder output: {x.shape}, feat len: {len(feat)}")

        return x, feat
