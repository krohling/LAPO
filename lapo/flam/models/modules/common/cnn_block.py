import torch.nn as nn

from flam.models.modules.common.activation import swish


class ResBlock(nn.Module):
    # modified from https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py
    def __init__(
        self,
        in_filters,
        out_filters,
        use_conv_shortcut=False,
        use_agn=False,
    ):
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn:     # agn is GroupNorm likewise, skip if there is agn before
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual
