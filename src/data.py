from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_names(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    return {
        "train": read_split(root / "train.txt"),
        "validation": read_split(root / "validation.txt"),
        "test": read_split(root / "test.txt"),
    }


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    mask = depth > 0
    if mask.any():
        depth = depth.copy()
        depth[mask] = (depth[mask] - depth[mask].min() + 0.001) / 2.0
    return depth


class Cloth3DDataset(Dataset):
    """RGB/depth pairs with the normalization used by D1 and D2."""

    def __init__(self, names, root, img_size=256, augment=False, imagenet=False, augmentation=None):
        self.names = list(names)
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.imagenet = imagenet
        self.augmentation = augmentation or {}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(self.root / "image" / f"{name}.jpg"))
        if image is None:
            raise FileNotFoundError(self.root / "image" / f"{name}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        depth = np.load(self.root / "depth" / f"{name}.npy").astype(np.float32)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            image, depth = self._augment(image, depth)
        depth = normalize_depth(depth)
        mask = depth > 0
        if self.imagenet:
            image = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        elif mask.any():
            image = image.copy() / 255.0
            for channel in range(3):
                values = image[:, :, channel][mask]
                image[:, :, channel] = (image[:, :, channel] - values.mean()) / (values.std() + 1e-5)
        else:
            image = image / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(depth).unsqueeze(0).float()

    def _augment(self, image, depth):
        if self.augmentation.get("hflip", True) and random.random() > 0.5:
            image, depth = image[:, ::-1].copy(), depth[:, ::-1].copy()
        if self.augmentation.get("rotate", True):
            angle = random.uniform(-15, 15)
            matrix = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (self.img_size, self.img_size))
            depth = cv2.warpAffine(depth, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
        return image, depth


class Cloth3DJointDataset(Cloth3DDataset):
    def __init__(self, names, root, joint_cache, img_size=256):
        super().__init__(names, root, img_size, imagenet=True)
        self.joint_cache = Path(joint_cache)

    def __getitem__(self, index):
        image, depth = super().__getitem__(index)
        path = self.joint_cache / f"{self.names[index]}.npz"
        if path.exists():
            cached = np.load(path)
            joints = cached["joints"].astype(np.float32)
            visible = cached["vis"].astype(np.float32)
        else:
            joints = np.zeros((14, 2), dtype=np.float32)
            visible = np.zeros(14, dtype=np.float32)
        return image, depth, torch.from_numpy(joints), torch.from_numpy(visible)
