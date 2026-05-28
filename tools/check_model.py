from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CSRNet, count_parameters


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test CSRNet forward pass.")
    parser.add_argument("--config", type=Path, default=Path("configs/csrnet_fdu.yaml"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--non-negative", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["train"]
    model_cfg = config.get("model", {})
    input_size = args.input_size if args.input_size is not None else int(train_cfg["input_size"])
    pretrained = (
        args.pretrained if args.pretrained is not None else bool(model_cfg.get("pretrained", False))
    )
    non_negative = (
        args.non_negative
        if args.non_negative is not None
        else bool(model_cfg.get("non_negative", True))
    )

    model = CSRNet(pretrained=pretrained, non_negative=non_negative)
    model.eval()
    total_params, trainable_params = count_parameters(model)

    x = torch.randn(args.batch_size, 3, input_size, input_size)
    with torch.no_grad():
        y = model(x)

    expected_size = input_size // 8
    expected_shape = (args.batch_size, 1, expected_size, expected_size)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(f"Unexpected output shape: got {tuple(y.shape)}, expected {expected_shape}")

    print(model)
    print(
        f"input={tuple(x.shape)} output={tuple(y.shape)} "
        f"pretrained={model.pretrained} total_params={total_params} "
        f"trainable_params={trainable_params} output_min={float(y.min()):.6f}"
    )


if __name__ == "__main__":
    main()
