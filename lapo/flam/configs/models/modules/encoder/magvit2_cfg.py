from dataclasses import dataclass, field
from typing import List


@dataclass
class Magvit2EncoderConfig:
    ch: int = 128
    num_res_blocks: int = 4
    ch_mult: List = field(default_factory=lambda: [1, 1, 2, 2, 4])  # num up-sample = len(ch_mult) - 1, each by 2
    z_channels: int = 8                                             # will be overwritten by the parent module
