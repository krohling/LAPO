import config
import data_loader
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from funcy import partition
from tensordict import TensorDict
from torch.distributions import Categorical

from config import IDMEncoderConfig, WMEncDecConfig

from einops import rearrange

from flam.utils.misc import compute_entropy_perplexity
from flam.models.modules.encoder import encoder_library
from flam.models.modules.decoder import decoder_library

ObsShapeType = tuple[int, int, int]  # (channels, height, width)


def merge_TC_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B, T*C, H, W)"""
    return x.view(x.shape[0], -1, *x.shape[3:])


class ResidualLayer(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_out_dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_out_dim, kernel_size, stride, padding),
        )

    def forward(self, x):
        return x + self.res_block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.Conv2d(in_depth, out_depth, 3, 1, padding=1)
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pool(self.res(self.norm(self.conv(x)))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_depth, out_depth, 2, 2)
        # maybe put bn before conv since that's where unet connections are catted
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(self.norm(self.conv(x))))


class WorldModel(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        print("***WM.__init__***")
        print(f"action_dim: {action_dim} in_depth: {in_depth} out_depth: {out_depth} base_size: {base_size}")

        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        print(f"down_sizes: {down_sizes}")
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                print(f"downsample {i}: {in_size} -> {out_size}")
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                print(f"downsample {i} (no pool): {in_size} -> {out_size}")
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            print(f"upsample {i}: {in_size} + {incoming} -> {out_size}")
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        print(f"action.shape: {action.shape}")
        print(f"state_seq.shape: {state_seq.shape}")
        state = merge_TC_dims(state_seq)

        print(f"state.shape: {state.shape}")
        _, _, h, w = state.shape
        action = action[:, :, None, None]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        print(f"action.shape: {action.shape}")
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1)

        xs = []
        print(f"x.shape: {x.shape}")
        for layer in self.down:
            x = layer(x)
            print(f"x.shape: {x.shape}")
            xs.append(x)

        print("done with downsample")
        xs[-1] = action

        for i, layer in enumerate(self.up):
            print(f"torch.cat([x, xs[-i - 1]].shape: {torch.cat([x, xs[-i - 1]], dim=1).shape}")
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1))
        return F.tanh(out) / 2

    def label(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, :-1]
        wm_targ = batch["obs"][:, -1]
        la = batch["la_q"]  # TODO: also allow using la(noq)
        batch["wm_pred"] = self(wm_in_seq, la)
        return F.mse_loss(batch["wm_pred"], wm_targ)


class EncDecWorldModel(nn.Module):
    """EncDec-based world model"""

    def __init__(self, 
        wm_cfg: WMEncDecConfig, 
        image_size: Union[int, Tuple[int, int]], 
        sub_traj_len: int, 
        action_dim: int=None
    ):
        super().__init__()
        encoder_type_cfg = wm_cfg.encoder_all[wm_cfg.encoder_type]
        decoder_type_cfg = wm_cfg.decoder_all[wm_cfg.decoder_type]

        self.encoder = encoder_library[wm_cfg.encoder_type](
            encoder_type_cfg, 
            image_size=image_size, 
            sub_traj_len=sub_traj_len, 
            action_dim=action_dim
        )
        self.decoder = decoder_library[wm_cfg.decoder_type](
            decoder_type_cfg,
            down_sizes=self.encoder.down_sizes,
            action_dim=action_dim
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """
        z, feat = self.encoder(state_seq, action)
        result = self.decoder(z, feat, action)

        return result

    def label(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, :-1]
        wm_targ = batch["obs"][:, -1]
        la = batch["la_q"]  # TODO: also allow using la(noq)
        batch["wm_pred"] = self(wm_in_seq, la)

        # from utils import save_seq_targ_pred_grids_labeled
        # save_seq_targ_pred_grids_labeled(
        #     wm_in_seq=wm_in_seq.detach(),           # (B, T, C, H, W)
        #     wm_targ=wm_targ.detach(),               # (B, C, H, W) or (B, T, C, H, W)
        #     wm_pred=batch["wm_pred"].detach().clamp(0, 1),      # (B, C, H, W) or (B, T, C, H, W)
        #     out_dir="debug_vis",
        #     prefix="step_{:07d}".format(getattr(self, "global_step", 0)),
        #     num_samples=4,
        #     repeat_target=False,
        #     repeat_pred_if_single=False
        # )

        return F.mse_loss(batch["wm_pred"], wm_targ)




def layer_init(layer, std=None, bias_const=0.0):
    if std is not None:
        std = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# based on https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ImpalaResidualBlock(self._out_channels)
        self.res_block1 = ImpalaResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self) -> tuple[int, int, int]:
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


def get_impala(
    shape: ObsShapeType,
    impala_cnn_scale: int,
    out_channels: tuple[int, ...],
    out_features: int,
) -> tuple[nn.Sequential, nn.Linear]:
    conv_stack = []
    for out_ch in out_channels:
        conv_seq = ConvSequence(shape, impala_cnn_scale * out_ch)
        shape = conv_seq.get_output_shape()
        conv_stack.append(conv_seq)
    conv_stack = nn.Sequential(*conv_stack, nn.Flatten(), nn.ReLU())
    fc = nn.Linear(in_features=np.prod(shape), out_features=out_features)
    return conv_stack, fc


class Policy(nn.Module):
    """IMPALA CNN-based policy"""

    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.conv_stack, self.fc = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )
        self.policy_head = nn.Linear(impala_features, action_dim)
        self.value_head = layer_init(nn.Linear(impala_features, 1), std=1)

    def forward(self, x):
        return self.policy_head(F.relu(self.fc(self.conv_stack(x))))

    def get_value(self, x):
        return self.value_head(F.relu(self.fc(self.conv_stack(x))))

    def get_action_and_value(self, x, action=None):
        hidden = F.relu(self.fc(self.conv_stack(x)))
        probs = Categorical(logits=self.policy_head(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(hidden)


class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        cfg: config.VQConfig,
        epsilon=1e-5,
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.epsilon = epsilon
        self.cfg = cfg
        self.batch_count = 0

        embedding = torch.zeros(cfg.num_codebooks, cfg.num_embs, cfg.emb_dim)
        embedding.uniform_(-1 / cfg.num_embs * 5, 1 / cfg.num_embs * 5)

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.ones(cfg.num_codebooks, cfg.num_embs))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward_2d(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(
            torch.sum(self.embedding**2, dim=2).unsqueeze(1)
            + torch.sum(x_flat**2, dim=2, keepdim=True),
            x_flat,
            self.embedding.transpose(1, 2),
            alpha=-2.0,
            beta=1.0,
        )
        indices = torch.argmin(distances, dim=-1)

        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(
            self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )
        quantized = quantized.view_as(x)

        if self.training:
            self.batch_count += 1
            if self.batch_count < self.cfg.warmup_steps:
                dw = torch.bmm(encodings.transpose(1, 2), x_flat) 
                count = torch.sum(encodings, dim=1).unsqueeze(-1) 
                used = count.squeeze(-1) > 0 
                new_embedding = self.embedding.clone()
                new_embedding[used] = (dw / (count + self.epsilon))[used]
                self.embedding = new_embedding
            else:
                self.ema_count = self.cfg.decay * self.ema_count + (
                    1 - self.cfg.decay
                ) * torch.sum(encodings, dim=1)

                # n = torch.sum(self.ema_count, dim=-1, keepdim=True)
                # self.ema_count = (
                #     (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
                # )
                dw = torch.bmm(encodings.transpose(1, 2), x_flat)
                self.ema_weight = (
                    self.cfg.decay * self.ema_weight + (1 - self.cfg.decay) * dw
                )

                self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.cfg.commitment_cost * e_latent_loss

        entropy, perplexity = compute_entropy_perplexity(
            rearrange(indices, "c b -> b c"),
            self.cfg.num_embs,
        )
        # print(f"entropy: {entropy}, perplexity: {perplexity}")
        # print("Codebook weights:", self.embedding)
        # print("Codebook counts:", self.ema_count)

        quantized = quantized.detach() + (x - x.detach())

        # avg_probs = torch.mean(encodings, dim=1)
        # perplexity = torch.exp(
        #     -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
        # )

        return (
            quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W),
            loss,
            entropy,
            perplexity,
            indices.view(N, B, H, W).permute(1, 0, 2, 3),
        )

    def forward(self, x):
        bs = len(x)
        x = x.view(
            bs,
            self.cfg.num_codebooks * self.cfg.emb_dim,
            self.cfg.num_discrete_latents,
            1,
        )

        z_q, loss, entropy, perplexity, indices = self.forward_2d(x)

        return (
            z_q.view(
                bs,
                self.cfg.num_codebooks
                * self.cfg.num_discrete_latents
                * self.cfg.emb_dim,
            ),
            loss,
            entropy,
            perplexity,
            indices,
        )

    def inds_to_z_q(self, indices):
        """look up quantization inds in embedding"""
        assert not self.training
        N, M, D = self.embedding.size()
        B, N_, H, W = indices.shape
        assert N == N_

        # N ... num_codebooks
        # M ... num_embs
        # D ... emb_dim
        # H ... num_discrete_latents (kinda)

        inds_flat = indices.permute(1, 0, 2, 3).reshape(N, B * H * W)
        quantized = torch.gather(
            self.embedding, 1, inds_flat.unsqueeze(-1).expand(-1, -1, D)
        )
        return (
            quantized.view(N, B, H, W, D).permute(1, 0, 4, 2, 3).reshape(B, N * D, H, W)
        )  # shape is (B, num_codebooks * emb_dim, num_discrete_latents, 1)


class IDM(nn.Module):
    """Quantized inverse dynamics model"""

    def __init__(
        self,
        vq_config: config.VQConfig,
        obs_shape: ObsShapeType,
        encoder_cfg: IDMEncoderConfig,
        image_size: Union[int, Tuple[int, int]], 
        sub_traj_len: int, 
        action_dim: int=None
    ):
        super().__init__()

        self.encoder_cfg = encoder_cfg
        encoder_type_cfg = encoder_cfg.encoder_all[encoder_cfg.encoder_type]

        if encoder_cfg.encoder_type == "default":
            # initialize impala CNN
            self.conv_stack, self.fc = get_impala(
                obs_shape, encoder_type_cfg.impala_scale, encoder_type_cfg.impala_channels, encoder_type_cfg.impala_features
            )
            self.policy_head = nn.Linear(encoder_type_cfg.impala_features, action_dim)
        else:
            self.encoder = encoder_library[encoder_cfg.encoder_type](encoder_type_cfg, image_size, sub_traj_len)

            encoder_features = self.encoder.N * self.encoder.D
            self.policy_head = nn.Linear(encoder_features, action_dim)

        # initialize quantizer
        self.vq = VQEmbeddingEMA(vq_config)

    def forward(self, x):
        """
        x.shape = (B, T, C, H, W)
        the IDM predicts the action between the last and second to last frames (T dim).
        """
        if self.encoder_cfg.encoder_type == "default":
            x = merge_TC_dims(x)
            la = self.policy_head(F.relu(self.fc(self.conv_stack(x))))
        else:
            x, _ = self.encoder(x)                         # -> (B, H_feat, W_feat, D)

            x = x.reshape((x.shape[0], np.prod(x.shape[1:])))  # -> (B, H_feat*W_feat*D)
            la = self.policy_head(F.relu(x))            # -> (B, action_dim)

        # print("Encoder output mean:", la.mean().item(), "std:", la.std().item())
        la_q, vq_loss, vq_entr, vq_perp, la_qinds = self.vq(la)

        action_dict = TensorDict(
            dict(
                la=la,
                la_q=la_q,
                la_qinds=la_qinds,
            ),
            batch_size=len(la),
        )

        stats = {
            "idm/vq_entropy": vq_entr,
            "idm/vq_perplexity": vq_perp,
            "idm/encoder_mean": la.mean(),
            "idm/encoder_std": la.std(),
        }

        return action_dict, vq_loss, stats

    def label(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, stats = self(batch["obs"])
        batch.update(action_td)
        return vq_loss, stats

    @torch.no_grad()
    def label_chunked(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            return self(data_loader.normalize_obs(batch["obs"].to(config.DEVICE)))[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data


if __name__ == "__main__":
    import torchinfo

    cfg = config.get(use_cli_args=False, override_args=["env.name=bossfight"])
    obs_shape = (3, 64, 64)
    policy = Policy(obs_shape, 15, 4)
    wm = WorldModel(cfg.model.la_dim, 3, 3, base_size=24)
    idm = IDM(cfg.model.vq, obs_shape, cfg.model.la_dim, 4)

    bs = 10
    obs = torch.randn(bs, 3, 64, 64)
    la = torch.randn(bs, cfg.model.la_dim)

    print("[WM]")
    print(torchinfo.summary(wm, input_data=(obs, la), depth=2))

    print("\n\n[Policy]")
    print(torchinfo.summary(policy, input_data=(obs,), depth=3))

    # torchinfo doesn't work with tensordict outputs
    orig_fwd = idm.forward
    idm.forward = lambda x: orig_fwd(x)[1]
    print("\n\n[IDM]")
    print(torchinfo.summary(idm, input_data=torch.cat([obs, obs]), depth=3))