from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import torch


def count_from_density(density: Any) -> float:
    """Return the count represented by a density map integral."""
    if isinstance(density, torch.Tensor):
        return float(density.detach().sum().item())
    return float(np.asarray(density, dtype=np.float64).sum())


def _count_arrays(
    pred_counts: Sequence[float],
    gt_counts: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_counts, dtype=np.float64)
    gt = np.asarray(gt_counts, dtype=np.float64)
    if pred.size == 0 or gt.size == 0:
        raise ValueError("metric inputs must not be empty")
    if pred.shape != gt.shape:
        raise ValueError(f"metric inputs must have the same shape, got {pred.shape} and {gt.shape}")
    return pred, gt


def mae(pred_counts: Sequence[float], gt_counts: Sequence[float]) -> float:
    """Mean absolute error for per-image counts."""
    pred, gt = _count_arrays(pred_counts, gt_counts)
    return float(np.mean(np.abs(pred - gt)))


def rmse(pred_counts: Sequence[float], gt_counts: Sequence[float]) -> float:
    """Root mean squared error for per-image counts."""
    pred, gt = _count_arrays(pred_counts, gt_counts)
    return float(math.sqrt(float(np.mean((pred - gt) ** 2))))

