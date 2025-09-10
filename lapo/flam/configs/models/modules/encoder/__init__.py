from dataclasses import dataclass, field

from flam.configs.models.modules.encoder.default_cfg import DefaultEncoderConfig
from flam.configs.models.modules.encoder.impala_cfg import ImpalaEncoderConfig
from flam.configs.models.modules.encoder.magvit2_cfg import Magvit2EncoderConfig


@dataclass
class EncoderConfig:
    default: DefaultEncoderConfig = field(default_factory=DefaultEncoderConfig)
    impala: ImpalaEncoderConfig = field(default_factory=ImpalaEncoderConfig)
    magvit2: Magvit2EncoderConfig = field(default_factory=Magvit2EncoderConfig)

    def __post_init__(self):
        self.configs = {
            "default": self.default,
            "impala": self.impala,
            "magvit2": self.magvit2,
        }
