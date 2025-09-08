"""
a wrapper over the DINOv2 model (https://github.com/facebookresearch/dinov2)
"""

from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T

from flam.configs.models.modules.encoder.dino_cfg import DinoEncoderConfig
from flam.utils.misc import pack_one, unpack_one


class DinoEncoder(nn.Module):
    def __init__(self, image_size: Union[int, Tuple[int, int]], encoder_cfg: DinoEncoderConfig):
        super().__init__()

        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.normalize = T.Normalize(mean=mean, std=std)

        model_name = encoder_cfg.model_name
        self.patch_size = int(model_name.split("-")[2])

        assert H % self.patch_size == 0, f"H % patch_size != 0, H = {H}, patch_size = {self.patch_size}"
        assert W % self.patch_size == 0, f"W % patch_size != 0, W = {W}, patch_size = {self.patch_size}"
        self.H_feat = image_size[0] // self.patch_size
        self.W_feat = image_size[1] // self.patch_size

        if dist.is_initialized():
            # avoid conflicts caused by multiple torch hub download
            rank = dist.get_rank()
            dist.barrier()
            if rank == 0:
                self.model = self.load_model(model_name)
            dist.barrier()
            if rank != 0:
                self.model = self.load_model(model_name)
            dist.barrier()
        else:
            self.model = self.load_model(model_name)

    def load_model(self, model_name):
        if model_name == "dinov2-vitb-14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.feat_dim = 768
        elif model_name == "dinov2-vitl-14":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.feat_dim = 1024
        else:
            raise ValueError(f"Encoder {model_name} not supported")

        for p in model.parameters():
            p.requires_grad = False

        return model

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

    @torch.no_grad()
    def forward(self, x):
        # :arg x:  (..., 3, H, W), normalized to [0, 1]
        # :return: (..., H_feat, W_feat, D)

        # preprocess
        x, ps = pack_one(x, "* d h w")                      # (..., 3, H, W) -> (B, 3, H, W)
        x = self.normalize(x)

        # dino
        self.model.eval()
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            x = blk(x)

        # remove cls token
        x = x[:, 1:]
        x = unpack_one(x, ps, "* h w d")                    # (B, H_feat, W_feat, D) -> (..., H_feat, W_feat, D)

        return x