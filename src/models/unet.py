import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = in_channels
        for feature in features[:-1]:
            self.encoders.append(ConvBlock(channels, feature))
            self.pools.append(nn.MaxPool2d(2))
            channels = feature
        self.bottleneck = ConvBlock(channels, features[-1])
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = features[-1]
        for feature in reversed(features[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(channels, feature, 2, stride=2))
            self.decoders.append(ConvBlock(feature * 2, feature))
            channels = feature
        self.head = nn.Sequential(nn.Conv2d(channels, out_channels, 1), nn.Sigmoid())

    def forward(self, inputs):
        skips = []
        features = inputs
        for encoder, pool in zip(self.encoders, self.pools):
            features = encoder(features)
            skips.append(features)
            features = pool(features)
        features = self.bottleneck(features)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            features = upconv(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat((skip, features), dim=1))
        return self.head(features)
