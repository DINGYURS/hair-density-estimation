from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SlidingCalibration:
    slope: float
    intercept: float
    source: str


def apply_calibration(pred_count: float, calibration: SlidingCalibration) -> float:
    return float(calibration.slope) * float(pred_count) + float(calibration.intercept)


def fit_linear_calibration(
    pred_counts: Iterable[float],
    gt_counts: Iterable[float],
    source: str,
) -> SlidingCalibration:
    pred = np.asarray(list(pred_counts), dtype=np.float64)
    gt = np.asarray(list(gt_counts), dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("pred_counts and gt_counts must have the same length")
    if pred.size < 2:
        raise ValueError("At least two prediction records are required for calibration")

    slope, intercept = np.polyfit(pred, gt, deg=1)
    return SlidingCalibration(
        slope=float(slope),
        intercept=float(intercept),
        source=str(source),
    )


def load_prediction_counts(path: str | Path) -> tuple[list[float], list[float]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows found: {csv_path}")
    gt_counts = [float(row["gt_count"]) for row in rows]
    pred_counts = [float(row["pred_count"]) for row in rows]
    return gt_counts, pred_counts


def save_calibration(calibration: SlidingCalibration, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(calibration), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> SlidingCalibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SlidingCalibration(
        slope=float(data["slope"]),
        intercept=float(data["intercept"]),
        source=str(data["source"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear calibration for sliding-window counts.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_counts, pred_counts = load_prediction_counts(args.predictions)
    calibration = fit_linear_calibration(
        pred_counts=pred_counts,
        gt_counts=gt_counts,
        source=str(args.predictions),
    )
    save_calibration(calibration, args.output)
    print(
        f"calibration: corrected_pred = {calibration.slope:.6f} * pred "
        f"+ {calibration.intercept:.6f}"
    )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
