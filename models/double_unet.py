import torch
import torch.nn as nn

from models.unet import UNetBilinear
from models.unet_subnetwork import UNetSubNetwork


class UNetDoublePoisson(nn.Module):
    """
    Splotowy model U-Net realizujący strategię dwóch niezależnych sieci:
    - u_net: generuje pole wektorowe u [B, 2, H, W]
    - h_net: generuje pole bezpieczeństwa h [B, 1, H, W]
    Wejście: siatka zajętości [B, 1, H, W]
    Wyjście: połączony rozkład [B, 3, H, W] (u_x, u_y, h)
    """

    def __init__(self):
        super(UNetDoublePoisson, self).__init__()
        self.u_net = UNetBilinear(in_channels=1, out_channels=2)
        self.h_net = UNetBilinear(in_channels=2, out_channels=1)

    def forward(self, x):
        u = self.u_net(x)
        h = self.h_net(u)
        return torch.cat([u, h], dim=1)
