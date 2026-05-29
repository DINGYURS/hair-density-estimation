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
from src.utils.voc_parser import VocObject


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def filter_objects_by_class(
    objects: list[VocObject],
    include_classes: set[str] | None = None,
    exclude_classes: set[str] | None = None,
) -> list[VocObject]:
    """Filter VOC objects by class names for ablation experiments."""
    if include_classes and exclude_classes:
        raise ValueError("include_classes and exclude_classes cannot both be set")

    if include_classes:
        return [obj for obj in objects if obj.class_name in include_classes]
    if exclude_classes:
        return [obj for obj in objects if obj.class_name not in exclude_classes]
    return list(objects)


def resize_image_and_points(
    image: np.ndarray,
    points: list[tuple[float, float]],
    output_size: int,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Resize the full image to a square and scale point annotations with it."""
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")

    height, width = image.shape[:2]
    scale_x = float(output_size) / float(width)
    scale_y = float(output_size) / float(height)
    resized = cv2.resize(image, (int(output_size), int(output_size)), interpolation=cv2.INTER_AREA)
    scaled_points = [
        (float(x) * scale_x, float(y) * scale_y)
        for x, y in points
        if 0.0 <= x < width and 0.0 <= y < height
    ]
    return resized, scaled_points


def _axis_window_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if length < crop_size:
        raise ValueError(f"image dimension {length} is smaller than crop_size {crop_size}")

    starts = list(range(0, max(length - crop_size + 1, 1), stride))
    last_start = length - crop_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_sliding_crop_boxes(
    height: int,
    width: int,
    crop_size: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    """Return the same fixed 512-style crop grid used by full-image sliding inference."""
    y_starts = _axis_window_starts(int(height), int(crop_size), int(stride))
    x_starts = _axis_window_starts(int(width), int(crop_size), int(stride))
    return [
        (left, top, left + int(crop_size), top + int(crop_size))
        for top in y_starts
        for left in x_starts
    ]


def select_sliding_crop_box(
    height: int,
    width: int,
    crop_size: int,
    stride: int,
    training: bool,
) -> tuple[tuple[int, int, int, int], int, int]:
    boxes = generate_sliding_crop_boxes(height, width, crop_size, stride)
    if training:
        index = random.randrange(len(boxes))
    else:
        image_center_x = float(width) / 2.0
        image_center_y = float(height) / 2.0
        index = min(
            range(len(boxes)),
            key=lambda item: (
                ((boxes[item][0] + boxes[item][2]) / 2.0 - image_center_x) ** 2
                + ((boxes[item][1] + boxes[item][3]) / 2.0 - image_center_y) ** 2
            ),
        )
    return boxes[index], int(index), len(boxes)


class FduDensityDataset(Dataset[dict[str, Any]]):
    """FDU bbox-center density dataset for CSRNet-style training."""

    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        input_size: int = 512,
        sigma: float = 4.0,
        sigma_mode: str = "fixed",
        adaptive_sigma_beta: float = 0.3,
        adaptive_sigma_min: float = 1.0,
        adaptive_sigma_max: float = 32.0,
        downsample: int = 8,
        training: bool = True,
        augment: bool = True,
        transform_mode: str = "crop",
        sliding_stride: int | None = None,
        include_classes: set[str] | list[str] | tuple[str, ...] | None = None,
        exclude_classes: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_file = self._resolve_split_file(split_file)
        self.input_size = int(input_size)
        self.sigma = float(sigma)
        self.sigma_mode = str(sigma_mode)
        self.adaptive_sigma_beta = float(adaptive_sigma_beta)
        self.adaptive_sigma_min = float(adaptive_sigma_min)
        self.adaptive_sigma_max = float(adaptive_sigma_max)
        self.downsample = int(downsample)
        self.training = bool(training)
        self.augment = bool(augment)
        self.transform_mode = str(transform_mode)
        self.sliding_stride = (
            max(1, self.input_size // 2) if sliding_stride is None else int(sliding_stride)
        )
        self.include_classes = set(include_classes) if include_classes else None
        self.exclude_classes = set(exclude_classes) if exclude_classes else None

        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if self.downsample <= 0:
            raise ValueError(f"downsample must be positive, got {downsample}")
        if self.sigma_mode not in {"fixed", "adaptive"}:
            raise ValueError(f"sigma_mode must be 'fixed' or 'adaptive', got {sigma_mode}")
        if self.sliding_stride <= 0:
            raise ValueError(f"sliding_stride must be positive, got {sliding_stride}")
        if self.transform_mode not in {"crop", "resize", "sliding_crop"}:
            raise ValueError(
                f"transform_mode must be 'crop', 'resize', or 'sliding_crop', got {transform_mode}"
            )
        if self.include_classes and self.exclude_classes:
            raise ValueError("include_classes and exclude_classes cannot both be set")

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
        objects = filter_objects_by_class(
            annotation.objects,
            include_classes=self.include_classes,
            exclude_classes=self.exclude_classes,
        )
        points = bbox_to_points(objects)

        if self.transform_mode == "resize":
            image, points = resize_image_and_points(image, points, self.input_size)
            crop_box = (0, 0, annotation.width, annotation.height)
            crop_index = -1
            crop_window_count = 1
        else:
            image, points, crop_box, crop_index, crop_window_count = self._crop_image_and_points(
                image, points
            )
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
            sigma_mode=self.sigma_mode,
            adaptive_sigma_beta=self.adaptive_sigma_beta,
            adaptive_sigma_min=self.adaptive_sigma_min,
            adaptive_sigma_max=self.adaptive_sigma_max,
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
            "crop_index": torch.tensor(int(crop_index), dtype=torch.int64),
            "crop_window_count": torch.tensor(int(crop_window_count), dtype=torch.int64),
            "crop_coverage": torch.tensor(
                float((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
                / float(annotation.width * annotation.height),
                dtype=torch.float32,
            ),
            "original_count": len(annotation.objects),
            "filtered_original_count": len(objects),
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
    ) -> tuple[np.ndarray, list[tuple[float, float]], tuple[int, int, int, int], int, int]:
        height, width = image.shape[:2]
        crop_size = min(self.input_size, height, width)

        if self.transform_mode == "sliding_crop":
            crop_box, crop_index, crop_window_count = select_sliding_crop_box(
                height=height,
                width=width,
                crop_size=crop_size,
                stride=self.sliding_stride,
                training=self.training,
            )
            left, top, right, bottom = crop_box
        elif self.training:
            left = random.randint(0, width - crop_size)
            top = random.randint(0, height - crop_size)
            crop_index = -1
            crop_window_count = 1
        else:
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            crop_index = -1
            crop_window_count = 1

        right = left + crop_size
        bottom = top + crop_size
        cropped = image[top:bottom, left:right].copy()
        cropped_points = [
            (x - left, y - top)
            for x, y in points
            if left <= x < right and top <= y < bottom
        ]
        return cropped, cropped_points, (left, top, right, bottom), crop_index, crop_window_count

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
