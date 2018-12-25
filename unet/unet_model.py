# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)    # First step of Contracting
        self.down1 = down(64, 128)    # Second step of Contracting
        self.down2 = down(128, 256)    # Third step of Contracting
        self.down3 = down(256, 512)    # Fourth step of Contracting
        self.down4 = down(512, 512)    # Bottleneck of U-Net
        self.up1 = up(1024, 256)    # First step of Expanding
        self.up2 = up(512, 128)    # Second step of Expanding
        self.up3 = up(256, 64)    # Third step of Expanding
        self.up4 = up(128, 64)    # Fourth step of Expanding
        # Output Conv layer with 1*1 filter
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # return F.sigmoid(x)
        return torch.sigmoid(x)
