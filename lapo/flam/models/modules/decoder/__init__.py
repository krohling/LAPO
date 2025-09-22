from flam.models.modules.decoder.lapo import LapoCnnDecoder
from flam.models.modules.decoder.magvit2 import Magvit2Decoder
from flam.models.modules.decoder.orig_lapo import LapoCnnDecoder as OrigLapoCnnDecoder
from flam.models.modules.decoder.orig_magvit2 import Magvit2Decoder as OrigMagvit2Decoder


decoder_library = {
    "lapo": LapoCnnDecoder,
    "magvit2": Magvit2Decoder,
    "orig_lapo": OrigLapoCnnDecoder,
    "orig_magvit2": OrigMagvit2Decoder,
}
