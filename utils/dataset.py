"""Duality/Falcon Offroad dataset loaders (train/val with Color_Images + Segmentation)"""

import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2


# Dataset label IDs -> contiguous train IDs [0..9]
ID_TO_TRAINID = {
    100: 0,     # Trees
    200: 1,     # Lush Bushes
    300: 2,     # Dry Grass
    500: 3,     # Dry Bushes
    550: 4,     # Ground Clutter
    600: 5,     # Flowers
    700: 6,     # Logs
    800: 7,     # Rocks
    7100: 8,    # Landscape / general ground
    10000: 9,   # Sky
}


def remap_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Convert raw mask IDs (100,200,...,10000) into train IDs (0..9).
    Unknown IDs become 0 by default (you can change this if needed).
    """
    out = np.zeros(mask_np.shape, dtype=np.int64)
    for raw_id, train_id in ID_TO_TRAINID.items():
        out[mask_np == raw_id] = train_id
    return out


class DualityDesertDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        super().__init__(root, transforms)

        self.img_dir = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")

        if not os.path.isdir(self.img_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"Expected folders:\n{self.img_dir}\n{self.mask_dir}"
            )

        # Pair by filename stem
        img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        self.image_paths = []
        self.mask_paths = []

        for f in img_files:
            stem = os.path.splitext(f)[0]
            # mask is usually PNG
            mask_path = os.path.join(self.mask_dir, stem + ".png")
            img_path = os.path.join(self.img_dir, f)
            if os.path.isfile(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        if len(self.image_paths) == 0:
            raise RuntimeError("No image/mask pairs found. Check filenames and extensions.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])

        mask_np = np.array(mask)
        # If mask is RGB, take one channel
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        mask_np = remap_mask(mask_np)
        mask_t = tv_tensors.Mask(mask_np, dtype=torch.long)

        if self.transforms:
            image, mask_t = self.transforms(image, mask_t)
        image = image.float()
        return image, mask_t


def get_dataloaders(data_dir: str, batch_size: int = 2) -> Dict[str, DataLoader]:
    transforms = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(513),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]
    )

    image_datasets = {
        "train": DualityDesertDataset(
            os.path.join(data_dir, "train"), transforms=transforms
        ),
        "valid": DualityDesertDataset(
            os.path.join(data_dir, "val"), transforms=transforms
        ),
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        ),
        "valid": DataLoader(
            image_datasets["valid"],
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        ),
    }

    return dataloaders
