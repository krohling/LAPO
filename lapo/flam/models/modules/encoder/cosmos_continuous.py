"""
A wrapper over the cosmos tokenizer (https://github.com/NVIDIA/Cosmos-Tokenizer)
"""

from typing import Union, Tuple

from einops import rearrange
from pathlib import Path

import torch.nn as nn

import cosmos_tokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks.configs import continuous_image

from flam.configs.models.modules.encoder.cosmos_continuous_cfg import CosmosContinuousEncoderConfig
from flam.utils.misc import pack_one, unpack_one


class CosmosContinuousEncoder(nn.Module):
    def __init__(self, image_size: Union[int, Tuple[int, int]], encoder_cfg: CosmosContinuousEncoderConfig):
        super().__init__()

        # Set default model name and resize dimensions if not provided in args
        self.model_name = encoder_cfg.model_name

        # Set checkpoint paths for encoder and decoder
        cosmos_path = Path(cosmos_tokenizer.__file__).parent.parent
        self.encoder_ckpt = f"{cosmos_path}/pretrained_ckpts/{self.model_name}/encoder.jit"

        # Initialize the ImageTokenizer with specified checkpoints
        self.tokenizer = ImageTokenizer(
            checkpoint_enc=self.encoder_ckpt,
            tokenizer_config=continuous_image,
            device="cuda",
            dtype="float32",
        )
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size

        down_sample_scale = encoder_cfg.down_sample_scale
        assert H % down_sample_scale == 0 and W % down_sample_scale == 0
        self.H_feat = H // down_sample_scale
        self.W_feat = W // down_sample_scale
        self.feat_dim = encoder_cfg.feat_dim

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

    def forward(self, x):
        # :arg x:  (..., 3, H, W), normalized to [0, 1]
        # :return: (..., H_feat, W_feat, D)

        # preprocess
        x, ps = pack_one(x, "* d h w")                      # (..., 3, H, W) -> (B, 3, H, W)

        x = self.tokenizer.encode(x)[0]

        # postprocess
        x = rearrange(x, "b d h w -> b h w d")
        x = unpack_one(x, ps, "* h w d")                    # (B, H_feat, W_feat, D) -> (..., H_feat, W_feat, D)

        assert x.shape[-3:] == (self.H_feat, self.W_feat, self.feat_dim)

        return x
