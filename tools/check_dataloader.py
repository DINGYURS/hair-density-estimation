from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.fdu_dataset import FduDensityDataset


SPLIT_KEYS = {
    "train": "train_split",
    "val": "val_split",
    "test": "test_split",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test FDU DataLoader.")
    parser.add_argument("--config", type=Path, default=Path("configs/csrnet_fdu.yaml"))
    parser.add_argument("--split", choices=sorted(SPLIT_KEYS), default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    train_cfg = config["train"]

    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    num_workers = (
        args.num_workers if args.num_workers is not None else int(train_cfg["num_workers"])
    )
    input_size = args.input_size if args.input_size is not None else int(train_cfg["input_size"])
    sigma = args.sigma if args.sigma is not None else float(train_cfg["sigma"])

    dataset = FduDensityDataset(
        data_root=Path(data_cfg["root"]),
        split_file=Path(data_cfg[SPLIT_KEYS[args.split]]),
        input_size=input_size,
        sigma=sigma,
        downsample=args.downsample,
        training=args.split == "train",
        augment=not args.no_augment,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.split == "train",
        num_workers=num_workers,
        pin_memory=False,
    )

    print(
        f"dataset={len(dataset)} split={args.split} batch_size={batch_size} "
        f"num_workers={num_workers} input_size={input_size} downsample={args.downsample}"
    )

    max_count_error = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        image = batch["image"]
        density = batch["density"]
        count = batch["count"]
        density_count = density.sum(dim=(1, 2, 3))
        count_error = torch.abs(density_count - count)
        max_count_error = max(max_count_error, float(count_error.max().item()))
        print(
            f"batch={batch_index} image={tuple(image.shape)} "
            f"density={tuple(density.shape)} count={count.tolist()} "
            f"density_sum={[round(v, 4) for v in density_count.tolist()]}"
        )
        if batch_index >= args.batches:
            break

    print(f"max_count_error={max_count_error:.6f}")


if __name__ == "__main__":
    main()
