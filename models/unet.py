import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    
class DoubleConvBi(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.middle = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        m = self.middle(self.pool(d2))

        u2 = self.up2(m)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)

# for 512x512 input grid
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        # Output layer (linear!)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        # Bottleneck
        m = self.middle(self.pool(d3))

        # Decoder
        u3 = self.up3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)

class UNetBilinear(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConvBi(in_channels, 32)
        self.down2 = DoubleConvBi(32, 64)
        self.down3 = DoubleConvBi(64, 128)
        self.down4 = DoubleConvBi(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = DoubleConvBi(256, 512)

        # Decoder (bilinear upsampling)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv4 = DoubleConvBi(512 + 256, 256)
        self.conv3 = DoubleConvBi(256 + 128, 128)
        self.conv2 = DoubleConvBi(128 + 64, 64)
        self.conv1 = DoubleConvBi(64 + 32, 32)

        # Output
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)              # 512
        d2 = self.down2(self.pool(d1))  # 256
        d3 = self.down3(self.pool(d2))  # 128
        d4 = self.down4(self.pool(d3))  # 64

        # Bottleneck
        m = self.middle(self.pool(d4))  # 32

        # Decoder
        u4 = self.up(m)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)

class UNetBilinearLite(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 16)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)
        self.down4 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = DoubleConv(128, 256)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv4 = DoubleConv(256 + 128, 128)
        self.conv3 = DoubleConv(128 + 64, 64)
        self.conv2 = DoubleConv(64 + 32, 32)
        self.conv1 = DoubleConv(32 + 16, 16)

        self.out = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        m = self.middle(self.pool(d4))

        u4 = self.up(m)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))

        u3 = self.up(u4)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(u1)