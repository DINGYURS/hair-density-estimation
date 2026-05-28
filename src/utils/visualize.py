from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.voc_parser import VocAnnotation


CLASS_COLORS = {
    "premium": (46, 204, 113),
    "single": (52, 152, 219),
    "undersize": (241, 196, 15),
    "abnormal": (231, 76, 60),
    "unknown": (149, 165, 166),
}


def draw_annotation(image: np.ndarray, annotation: VocAnnotation) -> np.ndarray:
    output = image.copy()
    for obj in annotation.objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj.bbox]
        color = CLASS_COLORS.get(obj.class_name, CLASS_COLORS["unknown"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = obj.class_name
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + baseline + 2)
        cv2.rectangle(
            output,
            (x1, label_y - label_size[1] - baseline - 2),
            (x1 + label_size[0] + 4, label_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 2, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"{annotation.image_id}  count={len(annotation.objects)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def save_annotation_preview(image_path: Path, annotation: VocAnnotation, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    output = draw_annotation(image, annotation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to write preview: {output_path}")


def density_to_heatmap(density: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    if density.ndim != 2:
        raise ValueError(f"density must be a 2D array, got shape {density.shape}")

    density_max = float(density.max()) if density.size else 0.0
    if density_max > 0.0:
        normalized = np.clip(density / density_max * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros(density.shape, dtype=np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    if output_size is not None:
        height, width = output_size
        heatmap = cv2.resize(heatmap, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0.0)


def draw_count_label(
    image: np.ndarray,
    image_id: str,
    gt_count: int,
    density_sum: float,
) -> np.ndarray:
    output = image.copy()
    text = f"{image_id}  gt={gt_count}  density_sum={density_sum:.3f}"
    cv2.putText(
        output,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
