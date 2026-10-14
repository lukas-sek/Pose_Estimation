import torch.nn as nn
import torch.nn.functional as F

from .deit import DeiTEncoder


class DepthJointDecoder(nn.Module):
    def __init__(self, in_channels, target_size=256, num_joints=14):
        super().__init__()
        channels = in_channels[-1]
        self.target_size = target_size
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.depth_head = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.joint_head = nn.Conv2d(64, num_joints, 1)

    def forward(self, features):
        features = F.interpolate(features[-1], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        features = self.trunk(features)
        return self.depth_head(features), self.joint_head(features)


class MultiTaskDepthModel(nn.Module):
    def __init__(self, patch_size=8, pretrained=True, img_size=256, num_joints=14):
        super().__init__()
        self.encoder = DeiTEncoder(patch_size=patch_size, pretrained=pretrained, img_size=img_size)
        self.decoder = DepthJointDecoder(self.encoder.out_channels, img_size, num_joints)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
