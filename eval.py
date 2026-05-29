from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import mae, rmse


LOGGER = logging.getLogger("eval")


@dataclass(frozen=True)
class PredictionRecord:
    image_id: str
    gt_count: float
    pred_count: float

    @property
    def abs_error(self) -> float:
        return abs(self.pred_count - self.gt_count)

    @property
    def squared_error(self) -> float:
        error = self.pred_count - self.gt_count
        return error * error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CSRNet checkpoint on FDU val/test split.")
    parser.add_argument("--config", type=Path, default=Path("configs/csrnet_fdu.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--sigma-mode", choices=("fixed", "adaptive"), default=None)
    parser.add_argument("--adaptive-sigma-beta", type=float, default=None)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--transform-mode", choices=("crop", "resize", "sliding_crop"), default=None)
    parser.add_argument("--include-class", action="append", default=None)
    parser.add_argument("--exclude-class", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --device cpu for local evaluation."
        )
    return device


def build_eval_dataset(
    config: dict[str, Any],
    split: str,
    input_size: int,
    sigma: float,
    sigma_mode: str,
    adaptive_sigma_beta: float,
    downsample: int,
    transform_mode: str,
    include_classes: list[str] | None,
    exclude_classes: list[str] | None,
) -> Any:
    from src.datasets import FduDensityDataset

    data_cfg = config["data"]
    split_key = f"{split}_split"
    return FduDensityDataset(
        data_root=Path(data_cfg["root"]),
        split_file=Path(data_cfg[split_key]),
        input_size=input_size,
        sigma=sigma,
        sigma_mode=sigma_mode,
        adaptive_sigma_beta=adaptive_sigma_beta,
        downsample=downsample,
        training=False,
        augment=False,
        transform_mode=transform_mode,
        sliding_stride=data_cfg.get("sliding_stride"),
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> Any:
    from src.models import CSRNet

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model_cfg = config.get("model", {})
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        model_cfg = {**checkpoint_config.get("model", {}), **model_cfg}

    model = CSRNet(
        pretrained=False,
        non_negative=bool(model_cfg.get("non_negative", True)),
    )
    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def _batch_pred_counts(pred_density: torch.Tensor) -> list[float]:
    counts = pred_density.detach().sum(dim=(1, 2, 3)).cpu().tolist()
    return [float(value) for value in counts]


@torch.no_grad()
def evaluate_model(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    model.eval()

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        gt_counts = batch["count"].detach().cpu().tolist()
        image_ids = batch["image_id"]

        pred_density = model(images)
        pred_counts = _batch_pred_counts(pred_density)

        for image_id, gt_count, pred_count in zip(image_ids, gt_counts, pred_counts, strict=True):
            records.append(
                PredictionRecord(
                    image_id=str(image_id),
                    gt_count=float(gt_count),
                    pred_count=float(pred_count),
                )
            )

        if max_batches is not None and batch_index >= max_batches:
            break

    if not records:
        raise RuntimeError("No evaluation samples were processed.")
    return records


def compute_summary(records: list[PredictionRecord]) -> dict[str, float]:
    if not records:
        raise ValueError("prediction records must not be empty")
    gt_counts = [record.gt_count for record in records]
    pred_counts = [record.pred_count for record in records]
    return {
        "samples": float(len(records)),
        "mae": mae(pred_counts, gt_counts),
        "rmse": rmse(pred_counts, gt_counts),
    }


def write_predictions_csv(records: list[PredictionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "gt_count", "pred_count", "abs_error", "squared_error"])
        for record in records:
            writer.writerow(
                [
                    record.image_id,
                    f"{record.gt_count:.6f}",
                    f"{record.pred_count:.6f}",
                    f"{record.abs_error:.6f}",
                    f"{record.squared_error:.6f}",
                ]
            )


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    train_cfg = config["train"]
    output_cfg = config.get("output", {})

    device_name = args.device if args.device is not None else str(train_cfg.get("device", "cpu"))
    device = resolve_device(device_name)
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    num_workers = (
        args.num_workers if args.num_workers is not None else int(train_cfg["num_workers"])
    )
    input_size = args.input_size if args.input_size is not None else int(train_cfg["input_size"])
    sigma = args.sigma if args.sigma is not None else float(train_cfg["sigma"])
    sigma_mode = (
        args.sigma_mode if args.sigma_mode is not None else str(train_cfg.get("sigma_mode", "fixed"))
    )
    adaptive_sigma_beta = (
        args.adaptive_sigma_beta
        if args.adaptive_sigma_beta is not None
        else float(train_cfg.get("adaptive_sigma_beta", 0.3))
    )
    data_cfg = config.get("data", {})
    transform_mode = (
        args.transform_mode
        if args.transform_mode is not None
        else str(data_cfg.get("transform_mode", "crop"))
    )
    include_classes = (
        args.include_class if args.include_class is not None else data_cfg.get("include_classes")
    )
    exclude_classes = (
        args.exclude_class if args.exclude_class is not None else data_cfg.get("exclude_classes")
    )
    output_dir = args.output_dir if args.output_dir is not None else Path(output_cfg.get("root", "outputs")) / "predictions"
    output_path = output_dir / f"{args.split}_predictions.csv"

    LOGGER.info(
        "config=%s checkpoint=%s split=%s device=%s batch_size=%d num_workers=%d "
        "input_size=%d sigma=%g sigma_mode=%s adaptive_sigma_beta=%g "
        "transform_mode=%s include_classes=%s exclude_classes=%s",
        args.config,
        args.checkpoint,
        args.split,
        device,
        batch_size,
        num_workers,
        input_size,
        sigma,
        sigma_mode,
        adaptive_sigma_beta,
        transform_mode,
        include_classes,
        exclude_classes,
    )

    dataset = build_eval_dataset(
        config=config,
        split=args.split,
        input_size=input_size,
        sigma=sigma,
        sigma_mode=sigma_mode,
        adaptive_sigma_beta=adaptive_sigma_beta,
        downsample=args.downsample,
        transform_mode=transform_mode,
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    records = evaluate_model(model, loader, device, max_batches=args.max_batches)
    summary = compute_summary(records)
    write_predictions_csv(records, output_path)

    LOGGER.info(
        "split=%s samples=%d mae=%.4f rmse=%.4f predictions=%s",
        args.split,
        int(summary["samples"]),
        summary["mae"],
        summary["rmse"],
        output_path,
    )
    print(
        f"split={args.split} samples={int(summary['samples'])} "
        f"mae={summary['mae']:.4f} rmse={summary['rmse']:.4f} "
        f"predictions={output_path}"
    )


if __name__ == "__main__":
    main()
