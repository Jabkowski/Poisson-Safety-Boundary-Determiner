import torch
import torch.nn as nn

from models.unet import UNetBilinear
from models.unet_subnetwork import UNetSubNetwork


def denormalize_pred_3ch(pred_3ch, normalization_stats):
    """
    Konwertuje wyjście modelu [ux_norm, uy_norm, h_norm] do skali fizycznej.
    """
    pred_phys = pred_3ch.clone()

    ux_mean = normalization_stats["ux"]["mean"]
    ux_std = normalization_stats["ux"]["std"]
    uy_mean = normalization_stats["uy"]["mean"]
    uy_std = normalization_stats["uy"]["std"]
    h_mean = normalization_stats["h"]["mean"]
    h_std = normalization_stats["h"]["std"]

    pred_phys[:, 0:1, :, :] = (
        pred_phys[:, 0:1, :, :] * ux_std + ux_mean
    )
    pred_phys[:, 1:2, :, :] = (
        pred_phys[:, 1:2, :, :] * uy_std + uy_mean
    )
    pred_phys[:, 2:3, :, :] = pred_phys[:, 2:3, :, :] * h_std + h_mean
    return pred_phys

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
