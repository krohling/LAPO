"""
Impala CNN encoder from https://arxiv.org/pdf/1802.01561
"""

from typing import Union, Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from flam.configs.models.modules.encoder.impala_cfg import ImpalaEncoderConfig
from flam.utils.misc import pack_one, unpack_one


# based on https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ImpalaResidualBlock(self._out_channels)
        self.res_block1 = ImpalaResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self) -> tuple[int, int, int]:
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

def merge_TC_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B, T*C, H, W)"""
    return x.view(x.shape[0], -1, *x.shape[3:])

# def unmerge_TC_dims(x: torch.Tensor):
#     """x.shape == (B, T*C, H, W) -> (B, T, C, H, W)"""
#     return x.view

class ImpalaCnnEncoder(nn.Module):
    def __init__(self, 
        encoder_cfg: ImpalaEncoderConfig, 
        image_size: Union[int, Tuple[int, int]], 
        sub_traj_len: int, 
        action_dim: int=None
    ):
        super().__init__()
        print("***ImpalaCnnEncoder.__init__***")

        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size

        ch_dim = (3*sub_traj_len) + (action_dim if action_dim is not None else 0)
        shape = [ch_dim, H, W]

        self.conv_stack = nn.ModuleList()
        self.down_sizes = []
        for out_ch in encoder_cfg.ch_mult:
            print(f"in_shape: {shape}, out_ch: {encoder_cfg.ch * out_ch}")
            conv_seq = ConvSequence(shape, encoder_cfg.ch * out_ch)
            shape = conv_seq.get_output_shape()
            self.down_sizes.append(shape[0])
            print(f"out_shape: {shape}")
            self.conv_stack.append(conv_seq)

        self.conv_out = nn.Conv2d(shape[0], encoder_cfg.z_channels, kernel_size=(1, 1))
        self.feat_dim = encoder_cfg.z_channels
        self.H_feat, self.W_feat = shape[1], shape[2]

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

        # preprocess
        # x, ps = pack_one(x, "* d h w")                      # (..., 3, H, W) -> (B, 3, H, W)
        print(f"x.shape: {x.shape}")
        x = merge_TC_dims(x)

        if action is not None:
            _, _, h, w = x.shape
            action = action[:, :, None, None]
            x = torch.cat([x, action.repeat(1, 1, h, w)], dim=1)
            print(f"x.shape after action cat: {x.shape}")

        xs = []
        for layer in self.conv_stack:
            x = layer(x)
            print(f"x shape after layer: {x.shape}")
            xs.append(x)

        x = self.conv_out(F.relu(x))
        print(f"x shape after conv_out: {x.shape}")

        # postprocess
        x = rearrange(x, "b d h w -> b h w d")
        # x = unpack_one(x, ps, "* h w d")                    # (B, H_feat, W_feat, D) -> (..., H_feat, W_feat, D)
        
        # x = x.reshape((B, T,) + x.shape[1:])
        print(f"final x.shape: {x.shape}")

        return x, xs
