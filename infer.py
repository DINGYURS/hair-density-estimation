from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import load_config, load_model_from_checkpoint, resolve_device
from src.datasets.fdu_dataset import IMAGENET_MEAN, IMAGENET_STD, filter_objects_by_class
from src.utils.density_map import bbox_to_points, resize_density_map_keep_count
from src.utils.metrics import count_from_density
from src.utils.visualize import density_to_heatmap, overlay_heatmap
from src.utils.voc_parser import parse_voc_xml


LOGGER = logging.getLogger("infer")


@dataclass(frozen=True)
class InferenceResult:
    image_id: str
    pred_count: float
    heatmap_path: Path
    overlay_path: Path
    gt_count: float | None = None

    @property
    def abs_error(self) -> float | None:
        if self.gt_count is None:
            return None
        return abs(self.pred_count - self.gt_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image CSRNet inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/csrnet_fdu.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--mode", choices=("crop", "sliding"), default="crop")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride in pixels. Defaults to input-size // 2 for smoother overlapping fusion.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Optional VOC XML path for GT count. Defaults to data_root/Annotations/<image_stem>.xml if present.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _center_crop_bgr(
    image_bgr: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    height, width = image_bgr.shape[:2]
    crop_size = min(int(input_size), height, width)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image_bgr[top:bottom, left:right].copy(), (left, top, right, bottom)


def _to_model_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


def preprocess_image(
    image_path: str | Path,
    input_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    cropped_bgr, _ = _center_crop_bgr(image_bgr, input_size)
    return _to_model_tensor(cropped_bgr), cropped_bgr


def _axis_window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if length < patch_size:
        raise ValueError(f"image dimension {length} is smaller than patch_size {patch_size}")

    starts = list(range(0, max(length - patch_size + 1, 1), stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_sliding_windows(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    """Return fixed-size windows that cover the full image including right/bottom edges."""
    y_starts = _axis_window_starts(int(height), int(patch_size), int(stride))
    x_starts = _axis_window_starts(int(width), int(patch_size), int(stride))
    return [
        (left, top, left + int(patch_size), top + int(patch_size))
        for top in y_starts
        for left in x_starts
    ]


def resolve_sliding_stride(requested_stride: int | None, input_size: int) -> int:
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    if requested_stride is not None:
        if requested_stride <= 0:
            raise ValueError(f"stride must be positive, got {requested_stride}")
        return int(requested_stride)
    return max(1, int(input_size) // 2)


def make_blend_weight(height: int, width: int, edge_floor: float = 0.05) -> np.ndarray:
    """Create a center-weighted 2D window to reduce patch-edge seam artifacts."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if not 0.0 <= float(edge_floor) < 1.0:
        raise ValueError(f"edge_floor must be in [0, 1), got {edge_floor}")

    y = np.hanning(max(3, int(height))).astype(np.float32)
    x = np.hanning(max(3, int(width))).astype(np.float32)
    if y.shape[0] != height:
        y = cv2.resize(y[:, None], (1, int(height)), interpolation=cv2.INTER_LINEAR).ravel()
    if x.shape[0] != width:
        x = cv2.resize(x[None, :], (int(width), 1), interpolation=cv2.INTER_LINEAR).ravel()

    weight = np.outer(y, x).astype(np.float32)
    max_value = float(weight.max())
    if max_value > 0.0:
        weight /= max_value
    return np.clip(weight, float(edge_floor), 1.0).astype(np.float32)


def stitch_window_densities(
    windows: list[tuple[int, int, int, int]],
    densities: list[np.ndarray],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Stitch patch density maps into a full-size density map with weighted overlap blending."""
    if len(windows) != len(densities):
        raise ValueError("windows and densities must have the same length")
    output_height, output_width = output_size
    full_density = np.zeros((int(output_height), int(output_width)), dtype=np.float32)
    weights = np.zeros_like(full_density)

    for window, density in zip(windows, densities, strict=True):
        left, top, right, bottom = window
        patch_height = bottom - top
        patch_width = right - left
        resized_density = resize_density_map_keep_count(
            density,
            output_size=(patch_height, patch_width),
        )
        blend_weight = make_blend_weight(patch_height, patch_width)
        full_density[top:bottom, left:right] += resized_density * blend_weight
        weights[top:bottom, left:right] += blend_weight

    covered = weights > 0.0
    if not bool(np.all(covered)):
        raise RuntimeError("Sliding windows did not cover the full image.")
    full_density[covered] /= weights[covered]
    return full_density


def _find_default_xml(image_path: Path, data_root: Path) -> Path | None:
    candidate = data_root / "Annotations" / f"{image_path.stem}.xml"
    return candidate if candidate.exists() else None


def load_crop_gt_count(
    image_path: str | Path,
    data_root: str | Path,
    input_size: int,
    xml_path: str | Path | None = None,
    include_classes: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_classes: list[str] | tuple[str, ...] | set[str] | None = None,
) -> float | None:
    image_path = Path(image_path)
    data_root = Path(data_root)
    resolved_xml = Path(xml_path) if xml_path is not None else _find_default_xml(image_path, data_root)
    if resolved_xml is None or not resolved_xml.exists():
        return None

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    _, (left, top, right, bottom) = _center_crop_bgr(image_bgr, input_size)

    annotation = parse_voc_xml(resolved_xml)
    objects = filter_objects_by_class(
        annotation.objects,
        include_classes=set(include_classes) if include_classes else None,
        exclude_classes=set(exclude_classes) if exclude_classes else None,
    )
    points = bbox_to_points(objects)
    crop_points = [
        (x, y)
        for x, y in points
        if left <= x < right and top <= y < bottom
    ]
    return float(len(crop_points))


def load_full_gt_count(
    image_path: str | Path,
    data_root: str | Path,
    xml_path: str | Path | None = None,
    include_classes: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_classes: list[str] | tuple[str, ...] | set[str] | None = None,
) -> float | None:
    image_path = Path(image_path)
    data_root = Path(data_root)
    resolved_xml = Path(xml_path) if xml_path is not None else _find_default_xml(image_path, data_root)
    if resolved_xml is None or not resolved_xml.exists():
        return None

    annotation = parse_voc_xml(resolved_xml)
    objects = filter_objects_by_class(
        annotation.objects,
        include_classes=set(include_classes) if include_classes else None,
        exclude_classes=set(exclude_classes) if exclude_classes else None,
    )
    return float(len(objects))


@torch.no_grad()
def predict_density(
    model: Any,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    pred_density = model(image_tensor.to(device))
    density_np = pred_density.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_count = count_from_density(density_np)
    return density_np, pred_count


@torch.no_grad()
def predict_sliding_density(
    model: Any,
    image_bgr: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
) -> tuple[np.ndarray, float, int]:
    height, width = image_bgr.shape[:2]
    windows = generate_sliding_windows(
        height=height,
        width=width,
        patch_size=patch_size,
        stride=stride,
    )
    densities: list[np.ndarray] = []
    model.eval()

    for left, top, right, bottom in windows:
        patch_bgr = image_bgr[top:bottom, left:right]
        patch_tensor = _to_model_tensor(patch_bgr)
        pred_density, _pred_count = predict_density(model, patch_tensor, device)
        densities.append(pred_density)

    full_density = stitch_window_densities(
        windows=windows,
        densities=densities,
        output_size=(height, width),
    )
    return full_density, count_from_density(full_density), len(windows)


def format_inference_label_lines(
    image_id: str,
    pred_count: float,
    gt_count: float | None,
) -> list[str]:
    if gt_count is None:
        return [image_id, f"pred={pred_count:.2f}"]
    return [
        image_id,
        f"gt={gt_count:.2f}  pred={pred_count:.2f}  err={abs(pred_count - gt_count):.2f}",
    ]


def _draw_text_with_outline(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _draw_inference_label(
    image_bgr: np.ndarray,
    image_id: str,
    pred_count: float,
    gt_count: float | None,
) -> np.ndarray:
    output = image_bgr.copy()
    for line_index, line in enumerate(format_inference_label_lines(image_id, pred_count, gt_count)):
        _draw_text_with_outline(output, line, (12, 28 + line_index * 28), 0.68)
    return output


def save_inference_visualizations(
    image_bgr: np.ndarray,
    pred_density: np.ndarray,
    output_dir: str | Path,
    image_id: str,
    pred_count: float,
    gt_count: float | None = None,
    alpha: float = 0.4,
) -> InferenceResult:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap = density_to_heatmap(pred_density, output_size=image_bgr.shape[:2])
    overlay = overlay_heatmap(image_bgr, heatmap, alpha=alpha)
    heatmap = _draw_inference_label(heatmap, image_id, pred_count, gt_count)
    overlay = _draw_inference_label(overlay, image_id, pred_count, gt_count)

    heatmap_path = output_dir / f"{image_id}_heatmap.jpg"
    overlay_path = output_dir / f"{image_id}_overlay.jpg"
    if not cv2.imwrite(str(heatmap_path), heatmap):
        raise RuntimeError(f"Failed to write heatmap: {heatmap_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write overlay: {overlay_path}")

    return InferenceResult(
        image_id=image_id,
        pred_count=float(pred_count),
        gt_count=None if gt_count is None else float(gt_count),
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    train_cfg = config["train"]
    data_cfg = config["data"]
    output_cfg = config.get("output", {})

    device_name = args.device if args.device is not None else str(train_cfg.get("device", "cpu"))
    device = resolve_device(device_name)
    input_size = args.input_size if args.input_size is not None else int(train_cfg["input_size"])
    stride = resolve_sliding_stride(args.stride, input_size)
    include_classes = data_cfg.get("include_classes")
    exclude_classes = data_cfg.get("exclude_classes")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(output_cfg.get("root", "outputs")) / "predictions" / "single"
    )

    LOGGER.info(
        "config=%s checkpoint=%s image=%s device=%s input_size=%d mode=%s stride=%d output_dir=%s",
        args.config,
        args.checkpoint,
        args.image,
        device,
        input_size,
        args.mode,
        stride,
        output_dir,
    )

    model = load_model_from_checkpoint(args.checkpoint, config, device)
    if args.mode == "sliding":
        image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {args.image}")
        display_bgr = image_bgr
        pred_density, pred_count, window_count = predict_sliding_density(
            model=model,
            image_bgr=image_bgr,
            device=device,
            patch_size=input_size,
            stride=stride,
        )
        gt_count = load_full_gt_count(
            image_path=args.image,
            data_root=Path(data_cfg["root"]),
            xml_path=args.xml,
            include_classes=include_classes,
            exclude_classes=exclude_classes,
        )
        LOGGER.info("sliding_windows=%d full_image_shape=%s", window_count, image_bgr.shape[:2])
    else:
        image_tensor, cropped_bgr = preprocess_image(args.image, input_size=input_size)
        display_bgr = cropped_bgr
        pred_density, pred_count = predict_density(model, image_tensor, device)
        gt_count = load_crop_gt_count(
            image_path=args.image,
            data_root=Path(data_cfg["root"]),
            input_size=input_size,
            xml_path=args.xml,
            include_classes=include_classes,
            exclude_classes=exclude_classes,
        )
    result = save_inference_visualizations(
        image_bgr=display_bgr,
        pred_density=pred_density,
        output_dir=output_dir,
        image_id=Path(args.image).stem,
        pred_count=pred_count,
        gt_count=gt_count,
        alpha=args.alpha,
    )

    if result.gt_count is None:
        print(
            f"image={result.image_id} pred_count={result.pred_count:.4f} "
            f"heatmap={result.heatmap_path} overlay={result.overlay_path}"
        )
    else:
        print(
            f"image={result.image_id} gt_count={result.gt_count:.4f} "
            f"pred_count={result.pred_count:.4f} abs_error={result.abs_error:.4f} "
            f"heatmap={result.heatmap_path} overlay={result.overlay_path}"
        )


if __name__ == "__main__":
    main()
