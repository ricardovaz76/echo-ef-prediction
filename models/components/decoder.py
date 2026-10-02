import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1        = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv1      = nn.Conv2d(256, 128, 3, padding=1)
        self.gate_conv1 = nn.Conv2d(256, 128, kernel_size=1)

        self.up2        = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2      = nn.Conv2d(128, 64, 3, padding=1)
        self.gate_conv2 = nn.Conv2d(128, 64, kernel_size=1)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, f2, f1):

        # First Up Sampling
        x = self.up1(x)

        # Ensure spatial shape is the same
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        # Gated Skip
        # Ensures only the relevant information from the skip connection is used
        f2_gate = torch.sigmoid(self.gate_conv1(torch.cat([x, f2], dim=1)))
        f2 = f2 * f2_gate
        x  = torch.cat([x, f2], dim=1)
        x  = F.relu(self.conv1(x))

        # Second Up Sampling
        x = self.up2(x)

        # Ensure spatial shape is the same
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)

        # Gated Skip
        # Ensures only the relevant information from the skip connection is used
        f1_gate = torch.sigmoid(self.gate_conv2(torch.cat([x, f1], dim=1)))
        f1 = f1 * f1_gate
        x  = torch.cat([x, f1], dim=1)
        x  = F.relu(self.conv2(x))

        # Final convolution
        x = self.out(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)

        return x
