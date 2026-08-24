import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def read_manifest(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["filepath"], int(row["label_idx"])))
    return rows


def base_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CropDiseaseDataset(Dataset):
    def __init__(self, manifest_path: Path, train: bool = False, transform=None):
        self.samples = read_manifest(manifest_path)
        self.transform = transform if transform is not None else base_transform(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        image = self.transform(image)
        return image, label


def apply_perturbation(image_bgr: np.ndarray, kind: str, level: int) -> np.ndarray:
    if level == 0:
        return image_bgr

    if kind == "noise":
        sigma = level * 12.0
        noise = np.random.normal(0, sigma, image_bgr.shape).astype(np.float32)
        out = image_bgr.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    if kind == "blur":
        ksize = level * 2 + 1
        return cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)

    if kind == "brightness":
        factor = 1.0 + level * 0.25
        out = image_bgr.astype(np.float32) * factor
        return np.clip(out, 0, 255).astype(np.uint8)

    if kind == "rotation":
        angle = level * 15
        h, w = image_bgr.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image_bgr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    raise ValueError(f"Unknown perturbation kind: {kind}")


def to_tensor_normalized(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std
