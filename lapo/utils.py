import config
import data_loader
import doy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import IDM, Policy, WorldModel, EncDecWorldModel
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader

from pathlib import Path
import torch
import torchvision
from torchvision.utils import make_grid

from pathlib import Path
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

@torch.no_grad()
def save_seq_targ_pred_grids_labeled(
    wm_in_seq: torch.Tensor,                # (B, T, C, H, W)  values in [0,1] or [-1,1] or uint8
    wm_targ: Optional[torch.Tensor],        # (B, C, H, W) or (B, T, C, H, W) or None
    wm_pred: Optional[torch.Tensor],        # (B, C, H, W) or (B, T-1, C, H, W) or (B, T, C, H, W) or None
    out_dir: str | Path,
    prefix: str = "wm_debug",
    num_samples: int = 4,
    repeat_target: bool = True,             # repeat single target frame across time
    repeat_pred_if_single: bool = True,     # repeat single pred frame across time
    align_pred: str = "seq",                # {'seq','targ','white','none'} for T-1 rollout alignment
    resize: tuple[int, int] | None = (256, 256),
    pad: int = 2,
    pad_value: float | str = "auto",        # 'auto' picks black/white for contrast
    show_labels: bool = True,
    row_labels: Optional[Iterable[str]] = None,  # e.g., ("in_seq","target","pred")
    font_path: Optional[str] = None,        # custom TTF path if you want
    font_size: int = 18,
    label_pad_px: int = 10,                 # inner padding inside label margin
) -> list[Path]:
    """
    Saves PNG grids comparing:
        Row 1: input sequence (wm_in_seq)
        Row 2: target (if provided)
        Row 3: predictions (if provided)
    with a labeled left margin.

    Returns list of saved file paths, one per batch item.
    """
    def to_01(x: torch.Tensor) -> torch.Tensor:
        # Accept float in [-1,1] or [0,1], or uint8 -> convert to [0,1]
        if not x.dtype.is_floating_point:
            x = x.float() / 255.0
        x = x.clamp(-1, 1)
        return (x + 1) / 2 if x.min() < 0 else x.clamp(0, 1)

    def ensure_5d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.unsqueeze(1) if x.dim() == 4 else x  # (B,C,H,W) -> (B,1,C,H,W)

    def crop_time(x: Optional[torch.Tensor], T: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x[:, :T]

    def compute_auto_pad_value(sample_seq_5d: torch.Tensor) -> float:
        # sample_seq_5d: (T, C, H, W) for a single batch item
        borders = torch.cat([
            sample_seq_5d[..., 0, :],        # top
            sample_seq_5d[..., -1, :],       # bottom
            sample_seq_5d[..., :, 0],        # left
            sample_seq_5d[..., :, -1],       # right
        ], dim=-1)
        med = torch.median(borders).item()
        return 1.0 - float(med > 0.5)        # if bright borders, use black; else white

    def pick_font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if path:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
        try:
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Normalize and ensure shapes ---
    x = ensure_5d(to_01(wm_in_seq.detach()))
    assert x is not None and x.dim() == 5, "wm_in_seq must be (B,T,C,H,W)"
    B, T_x, C, H, W = x.shape
    y = ensure_5d(to_01(wm_targ.detach())) if wm_targ is not None else None
    z = ensure_5d(to_01(wm_pred.detach())) if wm_pred is not None else None

    # --- Align time with sequence length T_x ---
    T = T_x

    # Targets
    if y is not None:
        T_y = y.shape[1]
        if T_y == 1 and repeat_target:
            y = y.repeat(1, T, 1, 1, 1)
            T_y = T
        # if T_y != T:
        #     T = min(T, T_y)

    # Predictions
    if z is not None:
        T_z = z.shape[1]
        if T_z == 1 and repeat_pred_if_single:
            z = z.repeat(1, T, 1, 1, 1)
            T_z = T
        if T_z == T_x - 1 and align_pred != "none":
            if align_pred == "seq":
                first = x[:, :1]
            elif align_pred == "targ" and y is not None:
                first = y[:, :1]
            elif align_pred == "white":
                first = torch.ones_like(z[:, :1])
            else:
                first = x[:, :1]
            z = torch.cat([first, z], dim=1)
            T_z = z.shape[1]
        # if T_z != T:
        #     T = min(T, T_z)

    # Final crop to common T
    x = crop_time(x, T)
    if y is not None:
        y = crop_time(y, T)
    if z is not None:
        z = crop_time(z, T)

    # Row labels (only for rows present)
    rows_present = ["in_seq"]
    if y is not None:
        rows_present.append("target")
    if z is not None:
        rows_present.append("pred")
    if row_labels is None:
        row_labels = rows_present
    else:
        # If custom labels given, truncate to match rows or map by count
        row_labels = list(row_labels)[: len(rows_present)]

    # Prepare font & colors
    font = pick_font(font_path, font_size)
    # We'll set text color as contrast to pad color, which we decide shortly.

    saved = []
    for i in range(min(num_samples, B)):
        row_tensors = [x[i]]
        if y is not None:
            row_tensors.append(y[i])
        if z is not None:
            row_tensors.append(z[i])

        imgs = torch.cat(row_tensors, dim=0)  # (rows*T, C, H, W)
        if resize is not None:
            imgs = torchvision.transforms.functional.resize(imgs, resize)

        # Determine pad value for contrast if needed
        pv = pad_value
        if pad_value == "auto":
            pv = compute_auto_pad_value(row_tensors[0])

        grid = make_grid(imgs, nrow=T, padding=pad, pad_value=float(pv))  # (C, Hgrid, Wgrid)

        # Convert to PIL for labeling
        pil_grid = torchvision.transforms.functional.to_pil_image(grid.cpu())
        Wgrid, Hgrid = pil_grid.size
        # Tile size after (optional) resize:
        tile_h = imgs.shape[-2]  # height of each frame
        # Compute left label margin width dynamically from text widths
        draw_tmp = ImageDraw.Draw(pil_grid)
        def text_size(s: str) -> tuple[int, int]:
            try:
                bbox = font.getbbox(s)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                return draw_tmp.textsize(s, font=font)

        if show_labels:
            max_label_w = max(text_size(lbl)[0] for lbl in row_labels) if row_labels else 0
            label_margin = max_label_w + 2 * label_pad_px
            # Create a new canvas with extra left margin; fill with pad color
            pad_rgb = int(round(255 * float(pv)))
            bg_color = (pad_rgb, pad_rgb, pad_rgb)
            labeled = Image.new("RGB", (Wgrid + label_margin, Hgrid), color=bg_color)
            labeled.paste(pil_grid, (label_margin, 0))

            draw = ImageDraw.Draw(labeled)
            # Pick text color contrasting with the margin color
            txt_color = (0, 0, 0) if pad_rgb > 127 else (255, 255, 255)
            stroke = (255, 255, 255) if txt_color == (0, 0, 0) else (0, 0, 0)

            # y position for each row center: top padding + row_index*(tile_h + pad) + tile_h/2
            rows_count = len(row_tensors)
            for r in range(rows_count):
                label = row_labels[r] if r < len(row_labels) else f"row{r}"
                tw, th = text_size(label)
                y_center = pad + r * (tile_h + pad) + tile_h // 2
                y_text = int(y_center - th / 2)
                x_text = label_margin - label_pad_px - tw  # right-align in margin
                draw.text((x_text, y_text), label, fill=txt_color, font=font,
                          stroke_width=2, stroke_fill=stroke)

            out_img = labeled
        else:
            out_img = pil_grid

        path = out_dir / f"{prefix}_b{i}_T{T}_rows{len(row_tensors)}.png"
        out_img.save(path)
        saved.append(path)

    return saved


@torch.no_grad()
def save_seq_and_target_grids(
    wm_in_seq: torch.Tensor,                # (B, T, C, H, W), values in [0,1] or [-1,1] or uint8
    wm_targ: torch.Tensor,                  # (B, C, H, W) or (B, T, C, H, W)
    out_dir: str | Path,
    prefix: str = "wm_debug",
    num_samples: int = 4,
    repeat_target: bool = True,             # repeat single target frame across time
    resize: tuple[int, int] | None = (256, 256),
    pad: int = 2,
    pad_value: float = 0.5,
) -> list[Path]:
    """
    Saves PNG grids comparing input sequence (row 1) vs target (row 2).
    Returns list of saved file paths.
    """
    def to_01(x: torch.Tensor) -> torch.Tensor:
        # Accepts float in [-1,1] or [0,1], or uint8 0..255
        if not x.dtype.is_floating_point:
            x = x.float() / 255.0
        x = x.clamp(-1, 1)
        x = (x + 1) / 2 if x.min() < 0 else x.clamp(0, 1)
        return x

    def ensure_5d(x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,1,C,H,W)
        return x.unsqueeze(1) if x.dim() == 4 else x

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = ensure_5d(to_01(wm_in_seq.detach()))
    y = ensure_5d(to_01(wm_targ.detach()))

    # Align time dimension
    B, T, C, H, W = x.shape
    if y.shape[1] == 1 and (repeat_target or y.shape[1] < T):
        y = y.repeat(1, T, 1, 1, 1)
    elif y.shape[1] != T:
        Tmin = min(T, y.shape[1])
        x, y = x[:, :Tmin], y[:, :Tmin]
        T = Tmin

    saved = []
    for i in range(min(num_samples, B)):
        # Stack rows: [seq frames..., target frames...]
        imgs = torch.cat([x[i], y[i]], dim=0)  # (2T, C, H, W)
        if resize is not None:
            imgs = torchvision.transforms.functional.resize(imgs, resize)

        grid = make_grid(imgs, nrow=T, padding=pad, pad_value=pad_value)  # (C,H,W) in [0,1]
        path = out_dir / f"{prefix}_b{i}_T{T}.png"
        torchvision.utils.save_image(grid.cpu(), str(path))
        saved.append(path)

    return saved



def obs_to_img(obs: Tensor) -> Tensor:
    return ((obs.permute(1, 2, 0) + 0.5) * 255).to(torch.uint8).numpy(force=True)


def create_decoder(in_dim, out_dim, device=config.DEVICE, hidden_sizes=(128, 128)):
    decoder = []
    in_size = h = in_dim
    for h in hidden_sizes:
        decoder.extend([nn.Linear(in_size, h), nn.ReLU()])
        in_size = h
    decoder.append(nn.Linear(h, out_dim))
    return nn.Sequential(*decoder).to(device)


def create_dynamics_models(
    model_cfg: config.ModelConfig, 
    feat_shape: int = (128, 8, 8),
    sub_traj_len: int = 2,
    state_dicts: dict | None = None,
) -> tuple[IDM, WorldModel]:
    # obs_depth = 3
    # idm_in_depth = obs_depth * (2 + config.get_add_time_horizon())
    # wm_in_depth = obs_depth * (1 + config.get_add_time_horizon())
    # wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        feat_shape,
        model_cfg.idm_encoder,
        sub_traj_len=sub_traj_len,
        action_dim=model_cfg.la_dim,
    ).to(config.DEVICE)
    
    # wm = WorldModel(
    #     model_cfg.la_dim,
    #     in_depth=wm_in_depth,
    #     out_depth=wm_out_depth,
    #     base_size=model_cfg.wm_scale,
    # ).to(config.DEVICE)
    
    wm = EncDecWorldModel(
        wm_cfg=model_cfg.wm_encdec,
        feat_shape=feat_shape,
        sub_traj_len=sub_traj_len-1,
        action_dim=model_cfg.la_dim,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_policy(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    policy = Policy(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)

    if state_dict is not None:
        policy.load_state_dict(state_dict, strict=strict_loading)

    return policy


def eval_latent_repr(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)


def train_decoder(
    data: TensorDict,  # tensordict with keys "la", "ta"
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(data["la"].shape[-1], TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)

    train_data, test_data = data[: len(data) // 2], data[len(data) // 2 :]

    dataloader = DataLoader(
        train_data,  # type: ignore
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    step = 0
    for i in range(epochs):
        for batch in dataloader:
            pred_ta = decoder(batch["la"])
            ta = batch["ta"][:, -2]
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            if step % 10 == 0:
                with torch.no_grad():
                    test_pred_ta = decoder(test_data["la"])
                    test_ta = test_data["ta"][:, -2]

                    logger(
                        step=i,
                        test_loss=F.cross_entropy(test_pred_ta, test_ta),
                        test_acc=(test_pred_ta.argmax(-1) == test_ta).float().mean(),
                    )
            step += 1

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=logger["test_acc"][-1],
        test_loss=logger["test_loss"][-1],
    )

    return decoder, metrics