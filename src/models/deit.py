import torch.nn as nn
import torch.nn.functional as F


class DeiTEncoder(nn.Module):
    def __init__(self, patch_size=16, pretrained=True, img_size=256, model_name="deit_small_patch16_224"):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, patch_size=patch_size, num_classes=0)
        self.out_channels = [self.backbone.embed_dim]

    def forward(self, inputs):
        tokens = self.backbone.forward_features(inputs)
        if tokens.ndim != 3:
            raise RuntimeError(f"Unexpected DeiT feature shape: {tokens.shape}")
        patches = tokens[:, 1:] if tokens.shape[1] > 1 else tokens
        side = int(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise RuntimeError(f"Patch tokens are not square: {patches.shape}")
        return [patches.reshape(inputs.shape[0], side, side, -1).permute(0, 3, 1, 2).contiguous()]


class BilinearDecoder(nn.Module):
    def __init__(self, in_channels, target_size=256):
        super().__init__()
        channels = in_channels[-1]
        self.target_size = target_size
        self.refine = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        )

    def forward(self, features):
        return self.refine(F.interpolate(features[-1], size=(self.target_size, self.target_size), mode="bilinear", align_corners=False))


class FPNDecoder(nn.Module):
    def __init__(self, in_channels, target_size=256, fpn_channels=128):
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Conv2d(in_channels[-1], fpn_channels, 1)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        )

    def forward(self, features):
        projected = self.proj(features[-1])
        return self.head(F.interpolate(projected, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False))


class DepthModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
