# modified from https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py

from einops import rearrange

from typing import List

import torch
import torch.nn as nn


from flam.configs.models.modules.decoder.lapo_cfg import LapoDecoderConfig
from flam.utils.misc import pack_one, unpack_one


class ResidualLayer(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_out_dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_out_dim, kernel_size, stride, padding),
        )

    def forward(self, x):
        return x + self.res_block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_depth, out_depth, 2, 2)
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(self.norm(self.conv(x))))


class LapoCnnDecoder(nn.Module):
    def __init__(self, decoder_cfg: LapoDecoderConfig, down_sizes: List[int], action_dim: int=128):
        super().__init__()
        # print("LapoCnnDecoder.__init__")
        ch = decoder_cfg.ch
        down_sizes = list(reversed(down_sizes))
        

        # print(f"down_sizes: {down_sizes}")

        # up-scaling
        up_sizes = [ch * mult for mult in decoder_cfg.ch_mult]
        # in_sizes = [decoder_cfg.z_channels] + up_sizes[:-1]
        out_sizes = up_sizes
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(zip(down_sizes, out_sizes)):
            # print(f"in_size: {in_size} out_size: {out_size}")
            incoming = action_dim if i == 0 else out_sizes[i - 1]
            self.up.append(UpsampleBlock(in_size+incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1], ch, kernel_size=3, stride=1, padding=1),
            ResidualLayer(ch, ch // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 3, 1, 1),
        )

    def forward(self, z, enc_features: List, action: torch.Tensor=None):
        # :arg z:  (..., H_feat, W_feat, D)
        # :return: (..., 3, H, W)

        # preprocess
        z, ps = pack_one(z, "* h w d")                      # (..., H, W, D) -> (B, H, W, D)
        z = rearrange(z, "b h w d -> b d h w")
        # print(f"z.shape after rearrange: {z.shape}")

        # concat action to the first feature map
        if action is not None:
            _, _, h, w = z.shape
            enc_features[-1] = action[:, :, None, None].repeat(1, 1, h, w)

        for i, layer in enumerate(self.up):
            
            # print(f"features[-i - 1].shape: {enc_features[-i - 1].shape}")
            z = torch.cat([z, enc_features[-i - 1]], dim=1)
            # print(f"z.shape: {z.shape}")
            z = layer(z)

        z = self.final_conv(z)

        # postprocess
        z = unpack_one(z, ps, "* d h w")                    # (B, 3, H, W) -> (..., 3, H, W)

        return z
