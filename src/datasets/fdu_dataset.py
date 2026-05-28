from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.density_map import bbox_to_points, make_density_map
from src.utils.voc_parser import parse_voc_xml


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FduDensityDataset(Dataset[dict[str, Any]]):
    """FDU bbox-center density dataset for CSRNet-style training."""

    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        input_size: int = 512,
        sigma: float = 4.0,
        downsample: int = 8,
        training: bool = True,
        augment: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_file = self._resolve_split_file(split_file)
        self.input_size = int(input_size)
        self.sigma = float(sigma)
        self.downsample = int(downsample)
        self.training = bool(training)
        self.augment = bool(augment)

        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if self.downsample <= 0:
            raise ValueError(f"downsample must be positive, got {downsample}")

        self.image_ids = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.image_ids:
            raise ValueError(f"Split file is empty: {self.split_file}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        annotation = parse_voc_xml(self.data_root / "Annotations" / f"{image_id}.xml")
        image_path = self.data_root / "Images" / annotation.filename
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = bbox_to_points(annotation.objects)

        image, points, crop_box = self._crop_image_and_points(image, points)
        if self.training and self.augment:
            image, points = self._augment_spatial(image, points)
            image = self._augment_color(image)

        crop_height, crop_width = image.shape[:2]
        density = make_density_map(
            points,
            height=crop_height,
            width=crop_width,
            sigma=self.sigma,
            downsample=self.downsample,
        )

        image_tensor = self._to_image_tensor(image)
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()
        count = torch.tensor(float(len(points)), dtype=torch.float32)

        return {
            "image": image_tensor,
            "density": density_tensor,
            "count": count,
            "image_id": image_id,
            "crop_box": crop_box,
            "original_count": len(annotation.objects),
        }

    def _resolve_split_file(self, split_file: str | Path) -> Path:
        split_path = Path(split_file)
        if split_path.is_absolute():
            return split_path
        return self.data_root / split_path

    def _crop_image_and_points(
        self,
        image: np.ndarray,
        points: list[tuple[float, float]],
    ) -> tuple[np.ndarray, list[tuple[float, float]], tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        crop_size = min(self.input_size, height, width)

        if self.training:
            left = random.randint(0, width - crop_size)
            top = random.randint(0, height - crop_size)
        else:
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2

        right = left + crop_size
        bottom = top + crop_size
        cropped = image[top:bottom, left:right].copy()
        cropped_points = [
            (x - left, y - top)
            for x, y in points
            if left <= x < right and top <= y < bottom
        ]
        return cropped, cropped_points, (left, top, right, bottom)

    def _augment_spatial(
        self,
        image: np.ndarray,
        points: list[tuple[float, float]],
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        height, width = image.shape[:2]
        transformed = points

        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            transformed = [(self._clamp_coord(width - x, width), y) for x, y in transformed]

        if random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1, :, :])
            transformed = [(x, self._clamp_coord(height - y, height)) for x, y in transformed]

        return image, transformed

    def _augment_color(self, image: np.ndarray) -> np.ndarray:
        contrast = random.uniform(0.9, 1.1)
        brightness = random.uniform(-12.0, 12.0)
        adjusted = image.astype(np.float32) * contrast + brightness
        return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _clamp_coord(value: float, upper: int) -> float:
        return min(max(float(value), 0.0), float(upper) - 1e-4)

    @staticmethod
    def _to_image_tensor(image: np.ndarray) -> torch.Tensor:
        image_float = image.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(image_float.transpose(2, 0, 1)).float()
