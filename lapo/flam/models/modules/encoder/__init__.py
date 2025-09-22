from flam.models.modules.encoder.impala import ImpalaCnnEncoder
from flam.models.modules.encoder.magvit2 import Magvit2Encoder
from flam.models.modules.encoder.orig_impala import ImpalaCnnEncoder as OrigImpalaCnnEncoder
from flam.models.modules.encoder.orig_magvit2 import Magvit2Encoder as OrigMagvit2Encoder

encoder_library = {
    "impala": ImpalaCnnEncoder,
    "magvit2": Magvit2Encoder,
    "orig_impala": OrigImpalaCnnEncoder,
    "orig_magvit2": OrigMagvit2Encoder,
}
