from flam.models.modules.encoder.impala import ImpalaCnnEncoder
from flam.models.modules.encoder.magvit2 import Magvit2Encoder

encoder_library = {
    "impala": ImpalaCnnEncoder,
    "magvit2": Magvit2Encoder,
}
