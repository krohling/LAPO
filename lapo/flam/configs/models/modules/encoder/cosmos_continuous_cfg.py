from dataclasses import dataclass
from typing import Literal


@dataclass
class CosmosContinuousEncoderConfig:
    # Literal[
    #     "Cosmos-0.1-Tokenizer-CI8x8", "Cosmos-0.1-Tokenizer-CI16x16",
    # ]
    model_name: str = "Cosmos-0.1-Tokenizer-CI16x16"

    def __post_init__(self):
        self.down_sample_scale = int(self.model_name.split("x")[-1])
        self.feat_dim = 16
