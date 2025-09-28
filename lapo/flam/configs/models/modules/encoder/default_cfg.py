from dataclasses import dataclass

@dataclass
class DefaultEncoderConfig:
    impala_scale: int = 4
    impala_channels: tuple[int, ...] = (16, 32, 32)
    impala_features: int = 256