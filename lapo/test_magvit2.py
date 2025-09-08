import math
import torch

from flam.configs.models.modules.encoder.magvit2_cfg import Magvit2EncoderConfig
from flam.models.modules.encoder.magvit2 import Magvit2Encoder
from flam.configs.models.modules.decoder.magvit2_cfg import Magvit2DecoderConfig
from flam.models.modules.decoder.magvit2 import Magvit2Decoder
# from models import WorldModel

A_DIM = 128
IMAGE_SIZE = 64
B, T, C, H, W = 5, 2, 3, IMAGE_SIZE, IMAGE_SIZE



# wm = WorldModel(
#     action_dim=A_DIM,
#     in_depth=(C*(T-1)),
#     out_depth=C,
#     base_size=24,
# )


encoder_cfg = Magvit2EncoderConfig(
    ch=128,
    num_res_blocks=2,
    ch_mult=[1, 1, 2, 2, 4],
    z_channels=8
)

decoder_cfg = Magvit2DecoderConfig(
    ch=128,
    num_res_blocks=4,
    ch_mult=[1, 1, 2, 2, 4],
    z_channels=8
)


encoder = Magvit2Encoder(
    encoder_cfg,
    IMAGE_SIZE,
    sub_traj_len=T,
    action_dim=A_DIM
)
decoder = Magvit2Decoder(decoder_cfg, encoder.down_sizes, A_DIM)

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