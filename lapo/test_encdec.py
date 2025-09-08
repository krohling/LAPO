import math
import torch

from flam.configs.models.modules.encoder.impala_cfg import ImpalaEncoderConfig
from flam.configs.models.modules.decoder.lapo_cfg import LapoDecoderConfig
from flam.models.modules.encoder.impala import ImpalaCnnEncoder
from flam.models.modules.decoder.lapo import LapoCnnDecoder

from models import WorldModel

A_DIM = 128
IMAGE_SIZE = 256
B, T, C, H, W = 5, 2, 3, IMAGE_SIZE, IMAGE_SIZE



wm = WorldModel(
    action_dim=A_DIM,
    in_depth=(C*(T-1)),
    out_depth=C,
    base_size=24,
)

base_size = 24
encode_steps = int(math.log(IMAGE_SIZE, 2))
ch_mult = [2**i for i in range(encode_steps)]
z_channels = ch_mult[-1]*base_size

encoder_cfg = ImpalaEncoderConfig(
    ch=base_size,
    z_channels=z_channels,
    ch_mult=ch_mult
)

decoder_cfg = LapoDecoderConfig(
    ch=base_size,
    z_channels=z_channels,
    ch_mult=list(reversed(ch_mult))
)


encoder = ImpalaCnnEncoder(encoder_cfg, H, T, A_DIM)
decoder = LapoCnnDecoder(decoder_cfg, down_sizes=encoder.down_sizes, action_dim=A_DIM)

a = torch.rand(B, A_DIM)
print("a:", a.shape)

x = torch.randn(B, T, C, H, W)
# wm(x, a)

# x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)

print("x:", x.shape)
z, feat = encoder(x, a)  # (F, B*T, H_feat, W_feat, D)

print(f"len(z):", len(z))
print("z.shape:", z.shape)
x_rec = decoder(z, feat, a)  # (B*T, C, H, W)
print("x_rec:", x_rec.shape)