from flam.models.modules.encoder.cosmos_continuous import CosmosContinuousEncoder
from flam.models.modules.encoder.dino import DinoEncoder
from flam.models.modules.encoder.impala import ImpalaCnnEncoder
from flam.models.modules.encoder.magvit2 import Magvit2Encoder

encoder_library = {
    "cosmos_continuous": CosmosContinuousEncoder,
    "dino": DinoEncoder,
    "impala": ImpalaCnnEncoder,
    "magvit2": Magvit2Encoder,
}
