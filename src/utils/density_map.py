from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import cv2
import numpy as np

from src.utils.voc_parser import VocObject


Point = tuple[float, float]


def bbox_to_points(objects: Iterable[VocObject | Mapping[str, Any]]) -> list[Point]:
    """Convert valid bbox objects to center-point annotations."""
    points: list[Point] = []
    for obj in objects:
        bbox = obj.bbox if isinstance(obj, VocObject) else obj["bbox"]
        x1, y1, x2, y2 = bbox
        points.append(((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))
    return points


def resize_density_map_keep_count(
    density: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Resize a density map while preserving its integral/count."""
    if density.ndim != 2:
        raise ValueError(f"density must be a 2D array, got shape {density.shape}")

    output_height, output_width = output_size
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")

    original_sum = float(density.sum(dtype=np.float64))
    resized = cv2.resize(
        density.astype(np.float32, copy=False),
        (int(output_width), int(output_height)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32, copy=False)

    resized_sum = float(resized.sum(dtype=np.float64))
    if original_sum != 0.0 and resized_sum != 0.0:
        resized *= original_sum / resized_sum
    return resized


def compute_adaptive_sigmas(
    points: Iterable[Point],
    beta: float = 0.3,
    min_sigma: float = 1.0,
    max_sigma: float = 32.0,
    fallback_sigma: float = 4.0,
) -> list[float]:
    """Compute per-point sigma from nearest-neighbor distance."""
    point_list = [(float(x), float(y)) for x, y in points]
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if min_sigma <= 0 or max_sigma <= 0:
        raise ValueError("min_sigma and max_sigma must be positive")
    if min_sigma > max_sigma:
        raise ValueError(f"min_sigma must be <= max_sigma, got {min_sigma}>{max_sigma}")
    if fallback_sigma <= 0:
        raise ValueError(f"fallback_sigma must be positive, got {fallback_sigma}")
    if len(point_list) <= 1:
        sigma = min(max(float(fallback_sigma), float(min_sigma)), float(max_sigma))
        return [sigma for _ in point_list]

    sigmas: list[float] = []
    for index, (x, y) in enumerate(point_list):
        nearest = min(
            math.hypot(x - other_x, y - other_y)
            for other_index, (other_x, other_y) in enumerate(point_list)
            if other_index != index
        )
        sigma = min(max(float(beta) * float(nearest), float(min_sigma)), float(max_sigma))
        sigmas.append(sigma)
    return sigmas


def make_density_map(
    points: Iterable[Point],
    height: int,
    width: int,
    sigma: float,
    downsample: int = 1,
    sigma_mode: str = "fixed",
    adaptive_sigma_beta: float = 0.3,
    adaptive_sigma_min: float = 1.0,
    adaptive_sigma_max: float = 32.0,
    adaptive_sigma_fallback: float | None = None,
) -> np.ndarray:
    """Create a Gaussian density map from point annotations.

    Each visible point contributes an integral of 1. If downsample > 1, the
    returned map is resized to height/downsample by width/downsample while
    preserving the total count.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}")
    if sigma_mode not in {"fixed", "adaptive"}:
        raise ValueError(f"sigma_mode must be 'fixed' or 'adaptive', got {sigma_mode}")

    point_list = [(float(x), float(y)) for x, y in points]
    if sigma_mode == "adaptive":
        point_sigmas = compute_adaptive_sigmas(
            point_list,
            beta=adaptive_sigma_beta,
            min_sigma=adaptive_sigma_min,
            max_sigma=adaptive_sigma_max,
            fallback_sigma=float(sigma if adaptive_sigma_fallback is None else adaptive_sigma_fallback),
        )
    else:
        point_sigmas = [float(sigma) for _ in point_list]

    density = np.zeros((int(height), int(width)), dtype=np.float32)

    for (x, y), point_sigma in zip(point_list, point_sigmas, strict=True):
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue

        radius = max(1, int(math.ceil(float(point_sigma) * 3.0)))
        center_x = int(round(x))
        center_y = int(round(y))
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(width, center_x + radius + 1)
        y2 = min(height, center_y + radius + 1)
        if x2 <= x1 or y2 <= y1:
            continue

        xs = np.arange(x1, x2, dtype=np.float32) - float(x)
        ys = np.arange(y1, y2, dtype=np.float32) - float(y)
        xx, yy = np.meshgrid(xs, ys)
        kernel = np.exp(
            -(xx * xx + yy * yy) / (2.0 * float(point_sigma) * float(point_sigma))
        ).astype(np.float32)

        kernel_sum = float(kernel.sum(dtype=np.float64))
        if kernel_sum > 0.0:
            density[y1:y2, x1:x2] += kernel / kernel_sum

    if downsample == 1:
        return density

    output_height = max(1, int(height) // int(downsample))
    output_width = max(1, int(width) // int(downsample))
    return resize_density_map_keep_count(density, (output_height, output_width))
