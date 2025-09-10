from dataclasses import dataclass, field

from flam.configs.models.modules.decoder.lapo_cfg import LapoDecoderConfig
from flam.configs.models.modules.decoder.magvit2_cfg import Magvit2DecoderConfig


@dataclass
class DecoderConfig:
    magvit2: Magvit2DecoderConfig = field(default_factory=Magvit2DecoderConfig)
    lapo: LapoDecoderConfig = field(default_factory=LapoDecoderConfig)

    def __post_init__(self):
        self.configs = {
            "magvit2": self.magvit2,
            "lapo": self.lapo,
        }
