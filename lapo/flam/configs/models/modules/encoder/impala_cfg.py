from dataclasses import dataclass, field
from typing import List


@dataclass
class ImpalaEncoderConfig:
    z_channels: int = 24*32                                             # will be overwritten by the parent module
    num_res_blocks: int = 4
    ch: int = 24
    ch_mult: List = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])     # num down-sample = len(ch_mult), each by 2
