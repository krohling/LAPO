from typing import Optional

from einops import rearrange

import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid

from flam.utils.plot.mix_mask import mix_images_with_masks


def plot_original_recon_rollout(
    images: torch.Tensor,                                   # (b, t, c, h, w), value in [0, 1]
    images_recon: Optional[torch.Tensor] = None,            # (b, t, c, h, w), value in [0, 1]
    images_rollout: Optional[torch.Tensor] = None,          # (b, t - 1, c, h, w), value in [0, 1]
    attn: Optional[torch.Tensor] = None,                    # (b, t, s, h_feat, w_feat), softmax value across s dimension
    num_images: int = 5,
):
    """
    Plot ground-truth, reconstructed, and rollout images in rows.
    Rollout images are for Latent Action Model / Dynamics Model logging.
    If attn is provided, visualize it on top of ground-truth images.
    """
    # preprocess
    images = images.clip(0, 1)
    assert images_recon is not None or images_rollout is not None, "images_recon and images_rollout cannot be both None"

    if images_recon is not None:
        images_recon = images_recon.clip(0, 1)

    if images_rollout is not None:
        images_rollout = images_rollout.clip(0, 1)
        assert images_rollout.shape[1] == images.shape[1] - 1, \
            f"images_rollout seq_len should be one less than images seq_len, but got {images_rollout.shape[1]} and {images.shape[1]}"

        first_image = torch.ones_like(images[:, :1])
        images_rollout = torch.cat([first_image, images_rollout], dim=1)                # (b, t, c, h, w)

    b, t, _, _, _ = images.shape

    # resize images to (256, 256) for better visibility
    num_images = min(num_images, b)
    images, images_recon, images_rollout = map(
        lambda x: (
            None
            if x is None
            else
            torchvision.transforms.functional.resize(
                rearrange(x[:num_images], "n t c h w -> (n t) c h w"),
                (256, 256),
            ).view(num_images, t, 3, 256, 256)
        ),
        [images, images_recon, images_rollout],
    )

    # if we have slot attention, visualize it on top of ground-truth images
    slot_masks = None
    if attn is not None:
        s = attn.shape[-3]

        # change softmax attention scores to hard attention
        attn = F.interpolate(
            rearrange(attn[:num_images], "n t s h w -> (n t) s h w"),
            (256, 256),
            mode="nearest",
        ).view(num_images, t, s, 256, 256)
        attn = attn.argmax(dim=-3)                                                      # (n, t, h, w)
        attn_mask = F.one_hot(attn, num_classes=s)                                      # (n, t, h, w, s)
        attn_mask = rearrange(attn_mask, "n t h w s -> n t s h w")

        images = mix_images_with_masks(
            rearrange(images, "n t c h w -> (n t) c h w"),
            rearrange(attn_mask, "n t s h w -> (n t) s h w"),
            alpha=0.3,
        )
        images = rearrange(images, "(n t) c h w -> n t c h w", n=num_images)

        slot_masks = mix_images_with_masks(
            rearrange(images, "n t c h w -> (n t) c h w"),
            rearrange(attn_mask, "n t s h w -> (n t) s h w"),
            alpha=1.0,
        )
        slot_masks = rearrange(slot_masks, "(n t) c h w -> n t c h w", n=num_images)

    # decide pad_value for better border visibility
    border_pixels = torch.cat([
        images[..., :, 0, :],
        images[..., :, -1, :],
        images[..., :, :, 0],
        images[..., :, :, -1]
    ], dim=-1)
    border_median = torch.median(border_pixels).item()

    # concatenate images to 3 rows and t columns
    output = []
    for i in range(num_images):
        image = [images[i]]

        if slot_masks is not None:
            image.append(slot_masks[i])

        if images_recon is not None:
            image.append(images_recon[i])

        if images_rollout is not None:
            image.append(images_rollout[i])

        image = torch.cat(image, dim=0)                                                 # (3 * t, c, h, w)

        image = make_grid(
            image,
            nrow=t,
            pad_value=1.0 - (border_median > 0.5),
        )                                                                               # (c, 3 * h, t * w)
        image = rearrange(image, "c h w -> h w c")
        output.append(image.cpu().numpy())

    return output