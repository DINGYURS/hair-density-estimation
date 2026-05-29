from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import FduDensityDataset
from src.models import CSRNet, count_parameters


LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSRNet on FDU bbox-center density maps.")
    parser.add_argument("--config", type=Path, default=Path("configs/csrnet_fdu.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--sigma-mode", choices=("fixed", "adaptive"), default=None)
    parser.add_argument("--adaptive-sigma-beta", type=float, default=None)
    parser.add_argument("--lambda-count", type=float, default=None)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--transform-mode", choices=("crop", "resize", "sliding_crop"), default=None)
    parser.add_argument("--include-class", action="append", default=None)
    parser.add_argument("--exclude-class", action="append", default=None)
    parser.add_argument("--freeze-frontend", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --device cpu for local smoke tests."
        )
    return device


def build_dataset(
    config: dict[str, Any],
    split: str,
    input_size: int,
    sigma: float,
    sigma_mode: str,
    adaptive_sigma_beta: float,
    downsample: int,
    augment: bool,
    transform_mode: str,
    include_classes: list[str] | None,
    exclude_classes: list[str] | None,
) -> FduDensityDataset:
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
        training=split == "train",
        augment=augment if split == "train" else False,
        transform_mode=transform_mode,
        sliding_stride=data_cfg.get("sliding_stride"),
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )


def build_loader(
    dataset: FduDensityDataset,
    batch_size: int,
    num_workers: int,
    training: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def density_count(density: torch.Tensor) -> torch.Tensor:
    return density.sum(dim=(1, 2, 3))


def compute_loss(
    pred_density: torch.Tensor,
    gt_density: torch.Tensor,
    lambda_count: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_loss = nn.functional.mse_loss(pred_density, gt_density)
    pred_count = density_count(pred_density)
    gt_count = density_count(gt_density)
    count_loss = torch.mean(torch.abs(pred_count - gt_count))
    loss = mse_loss + float(lambda_count) * count_loss
    return loss, mse_loss, count_loss


def apply_freeze_frontend(model: CSRNet, freeze_frontend: bool) -> None:
    """Freeze or unfreeze CSRNet frontend parameters."""
    for parameter in model.frontend.parameters():
        parameter.requires_grad = not freeze_frontend


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def run_train_epoch(
    model: CSRNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_count: float,
    epoch: int,
    log_interval: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_count_loss = 0.0
    seen_batches = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_density = model(images)
        loss, mse_loss, count_loss = compute_loss(pred_density, gt_density, lambda_count)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        total_count_loss += float(count_loss.item())
        seen_batches += 1

        if batch_index == 1 or batch_index % log_interval == 0:
            LOGGER.info(
                "epoch=%d train batch=%d loss=%.6f mse=%.6f count_loss=%.6f",
                epoch,
                batch_index,
                float(loss.item()),
                float(mse_loss.item()),
                float(count_loss.item()),
            )

        if max_batches is not None and batch_index >= max_batches:
            break

    if seen_batches == 0:
        raise RuntimeError("No training batches were processed.")

    return {
        "loss": total_loss / seen_batches,
        "mse": total_mse / seen_batches,
        "count_loss": total_count_loss / seen_batches,
    }


@torch.no_grad()
def run_validation(
    model: CSRNet,
    loader: DataLoader,
    device: torch.device,
    lambda_count: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_count_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_samples = 0
    seen_batches = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)
        gt_count = batch["count"].to(device, non_blocking=True)

        pred_density = model(images)
        loss, mse_loss, count_loss = compute_loss(pred_density, gt_density, lambda_count)
        pred_count = density_count(pred_density)
        abs_error = torch.abs(pred_count - gt_count)
        sq_error = (pred_count - gt_count) ** 2

        batch_size = int(images.shape[0])
        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        total_count_loss += float(count_loss.item())
        total_abs_error += float(abs_error.sum().item())
        total_sq_error += float(sq_error.sum().item())
        total_samples += batch_size
        seen_batches += 1

        if max_batches is not None and batch_index >= max_batches:
            break

    if seen_batches == 0 or total_samples == 0:
        raise RuntimeError("No validation batches were processed.")

    return {
        "loss": total_loss / seen_batches,
        "mse": total_mse / seen_batches,
        "count_loss": total_count_loss / seen_batches,
        "mae": total_abs_error / total_samples,
        "rmse": (total_sq_error / total_samples) ** 0.5,
    }


def save_checkpoint(
    path: Path,
    model: CSRNet,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best_mae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "best_mae": best_mae,
            "pretrained": model.pretrained,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    train_cfg = config["train"]
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    device_name = args.device if args.device is not None else str(train_cfg.get("device", "cpu"))
    device = resolve_device(device_name)
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    num_workers = (
        args.num_workers if args.num_workers is not None else int(train_cfg["num_workers"])
    )
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else float(train_cfg["learning_rate"])
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
    lambda_count = (
        args.lambda_count if args.lambda_count is not None else float(train_cfg["lambda_count"])
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
    pretrained = (
        args.pretrained if args.pretrained is not None else bool(model_cfg.get("pretrained", False))
    )
    freeze_frontend = (
        args.freeze_frontend
        if args.freeze_frontend is not None
        else bool(model_cfg.get("freeze_frontend", False))
    )
    output_dir = args.output_dir if args.output_dir is not None else Path(output_cfg.get("root", "outputs"))
    checkpoint_path = output_dir / "checkpoints" / "csrnet_best.pth"

    LOGGER.info(
        "config=%s device=%s batch_size=%d num_workers=%d epochs=%d lr=%g "
        "input_size=%d sigma=%g sigma_mode=%s adaptive_sigma_beta=%g "
        "lambda_count=%g transform_mode=%s include_classes=%s exclude_classes=%s "
        "pretrained=%s freeze_frontend=%s",
        args.config,
        device,
        batch_size,
        num_workers,
        epochs,
        learning_rate,
        input_size,
        sigma,
        sigma_mode,
        adaptive_sigma_beta,
        lambda_count,
        transform_mode,
        include_classes,
        exclude_classes,
        pretrained,
        freeze_frontend,
    )

    train_dataset = build_dataset(
        config,
        split="train",
        input_size=input_size,
        sigma=sigma,
        sigma_mode=sigma_mode,
        adaptive_sigma_beta=adaptive_sigma_beta,
        downsample=args.downsample,
        augment=not args.no_augment,
        transform_mode=str(data_cfg.get("train_transform_mode", transform_mode)),
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )
    val_dataset = build_dataset(
        config,
        split="val",
        input_size=input_size,
        sigma=sigma,
        sigma_mode=sigma_mode,
        adaptive_sigma_beta=adaptive_sigma_beta,
        downsample=args.downsample,
        augment=False,
        transform_mode=str(data_cfg.get("val_transform_mode", transform_mode)),
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )
    train_loader = build_loader(train_dataset, batch_size, num_workers, True, device)
    val_loader = build_loader(val_dataset, batch_size, num_workers, False, device)

    model = CSRNet(pretrained=pretrained, non_negative=bool(model_cfg.get("non_negative", True)))
    apply_freeze_frontend(model, freeze_frontend=freeze_frontend)
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    LOGGER.info(
        "model=CSRNet total_params=%d trainable_params=%d pretrained_loaded=%s freeze_frontend=%s",
        total_params,
        trainable_params,
        model.pretrained,
        freeze_frontend,
    )

    optimizer_parameters = trainable_parameters(model)
    if not optimizer_parameters:
        raise RuntimeError("No trainable parameters remain after applying model freeze settings.")
    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_count=lambda_count,
            epoch=epoch,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_validation(
            model=model,
            loader=val_loader,
            device=device,
            lambda_count=lambda_count,
            max_batches=args.max_val_batches,
        )

        LOGGER.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_mae=%.4f val_rmse=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["mae"],
            val_metrics["rmse"],
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_mae=best_mae,
            )
            LOGGER.info("saved best checkpoint: %s best_mae=%.4f", checkpoint_path, best_mae)

    LOGGER.info("training finished best_mae=%.4f checkpoint=%s", best_mae, checkpoint_path)


if __name__ == "__main__":
    main()
