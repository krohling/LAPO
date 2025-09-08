from dataclasses import dataclass, field

from flam.configs.models.modules.encoder.default_cfg import DefaultEncoderConfig
from flam.configs.models.modules.encoder.dino_cfg import DinoEncoderConfig
from flam.configs.models.modules.encoder.impala_cfg import ImpalaEncoderConfig
from flam.configs.models.modules.encoder.magvit2_cfg import Magvit2EncoderConfig
from flam.configs.models.modules.encoder.cosmos_continuous_cfg import CosmosContinuousEncoderConfig


@dataclass
class EncoderConfig:
    default: DefaultEncoderConfig = field(default_factory=DefaultEncoderConfig)
    cosmos_continuous: CosmosContinuousEncoderConfig = field(default_factory=CosmosContinuousEncoderConfig)
    dino: DinoEncoderConfig = field(default_factory=DinoEncoderConfig)
    impala: ImpalaEncoderConfig = field(default_factory=ImpalaEncoderConfig)
    magvit2: Magvit2EncoderConfig = field(default_factory=Magvit2EncoderConfig)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
            "cosmos_continuous": self.cosmos_continuous,
            "dino": self.dino,
            "impala": self.impala,
            "magvit2": self.magvit2,
        }
