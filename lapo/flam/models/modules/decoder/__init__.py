from flam.models.modules.decoder.cosmos_continuous import CosmosContinuousDecoder
from flam.models.modules.decoder.lapo import LapoCnnDecoder
from flam.models.modules.decoder.magvit2 import Magvit2Decoder


decoder_library = {
    "cosmos_continuous": CosmosContinuousDecoder,
    "lapo": LapoCnnDecoder,
    "magvit2": Magvit2Decoder,
}
