from dataclasses import dataclass
from typing import Literal


@dataclass
class CosmosContinuousDecoderConfig:
    # Literal[
    #     "Cosmos-0.1-Tokenizer-CI8x8", "Cosmos-0.1-Tokenizer-CI16x16",
    # ]
    model_name: str = "Cosmos-0.1-Tokenizer-CI16x16"
