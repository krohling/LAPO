from dataclasses import dataclass


@dataclass
class ImageLossConfig:
    # loss weights
    lpips_weight: float = 1.0
    eval_fid: bool = False
    eval_fvd: bool = False