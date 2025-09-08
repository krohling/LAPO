from dataclasses import dataclass
from typing import Literal


@dataclass
class DinoEncoderConfig:
    # Literal[
    #     "dinov2-vitb-14", "dinov2_vitl-14",
    # ]
    model_name: str = "dinov2-vitb-14"
