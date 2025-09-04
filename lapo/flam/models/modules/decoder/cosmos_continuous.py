"""
A wrapper over the cosmos tokenizer (https://github.com/NVIDIA/Cosmos-Tokenizer)
"""

from einops import rearrange
from pathlib import Path

import torch.nn as nn

import cosmos_tokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks.configs import continuous_image

from flam.configs.models.modules.decoder.cosmos_continuous_cfg import CosmosContinuousDecoderConfig
from flam.utils.misc import pack_one, unpack_one


class CosmosContinuousDecoder(nn.Module):
    def __init__(self, decoder_cfg: CosmosContinuousDecoderConfig):
        super().__init__()

        # Set default model name and resize dimensions if not provided in args
        self.model_name = decoder_cfg.model_name

        # Set checkpoint paths for decoder and decoder
        cosmos_path = Path(cosmos_tokenizer.__file__).parent.parent
        self.decoder_ckpt = f"{cosmos_path}/pretrained_ckpts/{self.model_name}/decoder.jit"

        # Initialize the ImageTokenizer with specified checkpoints
        self.tokenizer = ImageTokenizer(
            checkpoint_dec=self.decoder_ckpt,
            tokenizer_config=continuous_image,
            device="cuda",
            dtype="float32",
        )
        for p in self.tokenizer.parameters():
            p.requires_grad = False

    def forward(self, z):
        # :arg z:  (..., H_feat, W_feat, D)
        # :return: (..., 3, H, W)

        # preprocess
        z, ps = pack_one(z, "* h w d")                      # (..., H, W, D) -> (B, H, W, D)
        z = rearrange(z, "b h w d -> b d h w")

        x = self.tokenizer.decode(z)

        # postprocess
        x = unpack_one(x, ps, "* d h w")                    # (B, 3, H, W) -> (..., 3, H, W)

        return x
