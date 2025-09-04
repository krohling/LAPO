from dataclasses import dataclass, field
from typing import List


@dataclass
class LapoDecoderConfig:
    z_channels: int = 8                                             # will be overwritten by the parent module
    ch: int = 48
    ch_mult: List = field(default_factory=lambda: [16, 8, 4, 2])    # num up-sample = len(ch_mult), each by 2
