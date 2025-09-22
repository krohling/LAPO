import dataclasses
from einops import rearrange

import torch
import torch.nn as nn

# from flam.configs.models.modules.tokenizer.tokenizer_cfg import TokenizerConfig
from hdf5.hdf5_dataset import SubTrajectory
from flam.utils.misc import load_checkpoint, get_checkpoint_config
from flam.utils.plot.plot_comparison import plot_original_recon_rollout

from flam.models.modules.encoder import encoder_library
from flam.models.modules.decoder import decoder_library
from flam.models.modules.quantizer import quantizer_library

from flam.loss.image import ImageLoss

from types import SimpleNamespace
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


class Tokenizer(nn.Module):
    def __init__(
        self,
        tokenizer_load_path: str,
        sub_traj_len: int,
    ):
        super().__init__()
        self.tokenizer_cfg = get_checkpoint_config(tokenizer_load_path, "tokenizer_cfg")
        ns_tokenizer_cfg = dict_to_namespace(self.tokenizer_cfg)

        # self.tokenizer_cfg = tokenizer_cfg
        self.encoder = encoder_library[f"orig_{ns_tokenizer_cfg.encoder_type}"](ns_tokenizer_cfg.image_size, ns_tokenizer_cfg.encoder)
        self.decoder = decoder_library[f"orig_{ns_tokenizer_cfg.decoder_type}"](ns_tokenizer_cfg.decoder)
        self.quantizer = quantizer_library[ns_tokenizer_cfg.quantizer_type](ns_tokenizer_cfg.quantizer)

        # TODO: add GAN loss
        # self.image_loss = ImageLoss(tokenizer_cfg.image_loss).eval()

        self.load_checkpoint(tokenizer_load_path)

        self.image_logging_cache = None

    def encode(self, image: torch.Tensor):
        """
        args:
            x: (..., 3, h, w) tensor
        return:
            quant: (..., h_feat, w_feat, d) tensor
            quantizer_loss: scalar tensor
            quantizer_logging: dict[str, scalar tensor]
        """
        h = self.encoder(image)                                                 # (..., h_feat, w_feat, d)

        quant, indices, quantizer_loss, quantizer_logging = self.quantizer(h)   # (..., h_feat, w_feat, d), (..., h_feat, w_feat, c), scalar, dataclass

        quantizer_logging = quantizer_logging.to_dict()
        quantizer_logging["quantizer_loss"] = quantizer_loss
        quantizer_logging = {f"quantizer/{k}": v for k, v in quantizer_logging.items()}

        return quant, indices, quantizer_loss, quantizer_logging

    def decode(self, quant):
        """
        args:
            quant: (..., h_feat, w_feat, d) tensor
        return:
            dec: (..., 3, h, w) tensor
        """
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        """
        args:
            quant: (..., h_feat, w_feat, c) tensor
        return:
            dec: (..., 3, h, w) tensor
        """
        assert self.tokenizer_cfg.quantizer_type != "no_vq", "no_vq does not support decode_code"

        quant_b = self.quantizer.embed_code(code_b)                         # (..., h_feat, w_feat, d)
        dec = self.decode(quant_b)
        return dec

    # def forward(self, batch: SubTrajectory, is_last_batch: bool = False):
    #     image = batch.image                                                 # (..., 3, h, w)

    #     quant, _, quantizer_loss, quantizer_logging = self.encode(image)    # (..., h_feat, w_feat, d), scalar, dict
    #     image_recon = self.decode(quant)                                    # (..., 3, h, w)

    #     # loss + logging
    #     recon_loss, recon_logging = self.image_loss(image_recon, image)     # scalar, dict
    #     loss = recon_loss + quantizer_loss

    #     logging = {**recon_logging, **quantizer_logging}

    #     # image logging
    #     if self.training:
    #         update_image_logging_cache = True
    #     else:
    #         update_image_logging_cache = self.image_logging_cache is None

    #     if update_image_logging_cache:
    #         self.image_logging_cache = {
    #             "image": image,
    #             "image_recon": image_recon,
    #         }

    #     return loss, logging

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantizer.parameters()),
            lr=self.tokenizer_cfg.learning_rate,
            betas=(0.5, 0.9),
        )
        return opt_gen

    # ============ Save and Load ============
    def image_logging(self):
        image_logging = {}
        if self.image_logging_cache is not None:
            image = self.image_logging_cache["image"]                                           # (b, 1, 3, h, w)
            image_recon = self.image_logging_cache["image_recon"]                               # (b, 1, 3, h, w)

            # logging image with (num_cols) columns
            num_cols = 8
            b = image.shape[0]
            b = num_cols * (b // num_cols)
            image = rearrange(image[:b], "(n t) 1 c h w -> n t c h w", t=num_cols)              # (b // 8, 8, 3, h, w)
            image_recon = rearrange(image_recon[:b], "(n t) 1 c h w -> n t c h w", t=num_cols)  # (b // 8, 8, 3, h, w)

            image_logging["comparison"] = plot_original_recon_rollout(
                image,
                image_recon,
            )
        self.image_logging_cache = None
        return image_logging

    def load_checkpoint(self, checkpoint_path):
        load_checkpoint(
            self,
            checkpoint_path,
            cfg_key="tokenizer_cfg",
            cfg_exclude_keys=(
                "sub_traj_len",
                "image_loss",
                "lpips_weight",
                "batch_size",
                "learning_rate",
                "grad_norm_clip",
                "load_path",
                "encoder_all",
                "decoder_all",
                "quantizer_all",
            ),
            only_check_common_keys=True,
            strict=False,               # vgg weights don't need to be loaded
        )

    def save_dict(self):
        ignored_module_prefix = ["inception_model", "lpips_loss"]
        state_dict = {
            k: v
            for k, v in self.state_dict().items()
            if not any([module_prefix in k for module_prefix in ignored_module_prefix])
        }
        return {
            "model": state_dict,
            "tokenizer_cfg": dataclasses.asdict(self.tokenizer_cfg),
        }