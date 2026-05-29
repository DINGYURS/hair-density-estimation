from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import (
    PredictionRecord,
    compute_summary,
    load_config,
    load_model_from_checkpoint,
    resolve_device,
    setup_logging,
    write_predictions_csv,
)
from infer import load_full_gt_count, predict_sliding_density, resolve_sliding_stride


LOGGER = logging.getLogger("eval_sliding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CSRNet checkpoint with full-image sliding-window inference."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/csrnet_fdu.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride in pixels. Defaults to input-size // 2.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def read_split_image_ids(split_path: str | Path) -> list[str]:
    path = Path(split_path)
    image_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [image_id for image_id in image_ids if image_id]


def resolve_sliding_output_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}_sliding_predictions.csv"


def _image_path_for_id(data_root: Path, image_id: str) -> Path:
    return data_root / "Images" / f"{image_id}.jpg"


@torch.no_grad()
def evaluate_sliding_image(
    model: Any,
    image_id: str,
    data_root: Path,
    device: torch.device,
    input_size: int,
    stride: int,
    include_classes: list[str] | None,
    exclude_classes: list[str] | None,
) -> PredictionRecord:
    image_path = _image_path_for_id(data_root, image_id)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    _density, pred_count, window_count = predict_sliding_density(
        model=model,
        image_bgr=image_bgr,
        device=device,
        patch_size=input_size,
        stride=stride,
    )
    gt_count = load_full_gt_count(
        image_path=image_path,
        data_root=data_root,
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )
    if gt_count is None:
        raise FileNotFoundError(f"Failed to find XML annotation for image: {image_path}")

    LOGGER.info(
        "image_id=%s windows=%d gt=%.4f pred=%.4f abs_error=%.4f",
        image_id,
        window_count,
        gt_count,
        pred_count,
        abs(float(pred_count) - float(gt_count)),
    )
    return PredictionRecord(
        image_id=image_id,
        gt_count=float(gt_count),
        pred_count=float(pred_count),
    )


def evaluate_sliding_split(
    model: Any,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    input_size: int,
    stride: int,
    max_images: int | None = None,
) -> list[PredictionRecord]:
    data_cfg = config["data"]
    data_root = Path(data_cfg["root"])
    split_path = data_root / Path(data_cfg[f"{split}_split"])
    image_ids = read_split_image_ids(split_path)
    if max_images is not None:
        image_ids = image_ids[: int(max_images)]
    if not image_ids:
        raise RuntimeError(f"No images found for split: {split}")

    include_classes = data_cfg.get("include_classes")
    exclude_classes = data_cfg.get("exclude_classes")
    records: list[PredictionRecord] = []
    for index, image_id in enumerate(image_ids, start=1):
        LOGGER.info("evaluating %d/%d image_id=%s", index, len(image_ids), image_id)
        records.append(
            evaluate_sliding_image(
                model=model,
                image_id=image_id,
                data_root=data_root,
                device=device,
                input_size=input_size,
                stride=stride,
                include_classes=include_classes,
                exclude_classes=exclude_classes,
            )
        )
    return records


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    train_cfg = config["train"]
    output_cfg = config.get("output", {})

    device_name = args.device if args.device is not None else str(train_cfg.get("device", "cpu"))
    device = resolve_device(device_name)
    input_size = args.input_size if args.input_size is not None else int(train_cfg["input_size"])
    stride = resolve_sliding_stride(args.stride, input_size)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(output_cfg.get("root", "outputs")) / "predictions"
    )
    output_path = resolve_sliding_output_path(output_dir, args.split)

    LOGGER.info(
        "config=%s checkpoint=%s split=%s device=%s input_size=%d stride=%d output=%s",
        args.config,
        args.checkpoint,
        args.split,
        device,
        input_size,
        stride,
        output_path,
    )

    model = load_model_from_checkpoint(args.checkpoint, config, device)
    records = evaluate_sliding_split(
        model=model,
        config=config,
        split=args.split,
        device=device,
        input_size=input_size,
        stride=stride,
        max_images=args.max_images,
    )
    summary = compute_summary(records)
    write_predictions_csv(records, output_path)
    LOGGER.info(
        "sliding split=%s samples=%d mae=%.4f rmse=%.4f predictions=%s",
        args.split,
        int(summary["samples"]),
        summary["mae"],
        summary["rmse"],
        output_path,
    )
    print(
        f"sliding split={args.split} samples={int(summary['samples'])} "
        f"mae={summary['mae']:.4f} rmse={summary['rmse']:.4f} "
        f"predictions={output_path}"
    )


if __name__ == "__main__":
    main()
