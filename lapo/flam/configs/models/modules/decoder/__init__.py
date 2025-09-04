from dataclasses import dataclass, field

from flam.configs.models.modules.decoder.cosmos_continuous_cfg import CosmosContinuousDecoderConfig
from flam.configs.models.modules.decoder.lapo_cfg import LapoDecoderConfig
from flam.configs.models.modules.decoder.magvit2_cfg import Magvit2DecoderConfig


@dataclass
class DecoderConfig:
    cosmos_continuous: CosmosContinuousDecoderConfig = field(default_factory=CosmosContinuousDecoderConfig)
    magvit2: Magvit2DecoderConfig = field(default_factory=Magvit2DecoderConfig)
    lapo: LapoDecoderConfig = field(default_factory=LapoDecoderConfig)

    def __post_init__(self):
        self.configs = {
            "cosmos_continuous": self.cosmos_continuous,
            "magvit2": self.magvit2,
            "lapo": self.lapo,
        }
