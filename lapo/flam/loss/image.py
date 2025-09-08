import torch
import torch.nn as nn
import torch.nn.functional as F

from flam.loss.lpips import LpipsLoss


class ImageLoss(nn.Module):
    def __init__(self, image_loss_cfg):
        super().__init__()
        self.lpips_weight = image_loss_cfg.lpips_weight
        self.eval_fid = image_loss_cfg.eval_fid
        self.eval_fvd = image_loss_cfg.eval_fvd

        if self.lpips_weight > 0:
            self.lpips_loss = LpipsLoss()
            self.lpips_loss.eval()

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)

        image_loss = mse
        image_logging = {
            "pixel_l1": l1_loss,
            "pixel_mse": mse,
            "psnr": -10 * torch.log10(mse),     # MAX is 1.0 so we ignore it
        }

        if self.lpips_weight > 0:
            self.lpips_loss.eval()
            lpips_loss = self.lpips_loss(pred, target).mean()
            image_loss = image_loss + self.lpips_weight * lpips_loss
            image_logging["lpips_loss"] = lpips_loss

        image_logging["image_loss"] = image_loss

        return image_loss, image_logging