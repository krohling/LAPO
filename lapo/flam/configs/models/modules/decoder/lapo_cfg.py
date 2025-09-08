from dataclasses import dataclass, field
from typing import List


@dataclass
class LapoDecoderConfig:
    z_channels: int = 24*32                                             # will be overwritten by the parent module
    ch: int = 24
    ch_mult: List = field(default_factory=lambda: [32, 16, 8, 4, 2, 1, 1])    # num up-sample = len(ch_mult), each by 2
