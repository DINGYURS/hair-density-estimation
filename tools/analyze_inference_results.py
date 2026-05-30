from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class InferenceRecord:
    image_id: str
    gt_count: float
    pred_count: float
    abs_error: float
    corrected_pred_count: float | None = None
    corrected_abs_error: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze full-image inference prediction CSV and generate report figures."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default="FDU full-image inference analysis")
    parser.add_argument("--worst-k", type=int, default=30)
    return parser.parse_args()


def load_records(path: Path) -> list[InferenceRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows found: {path}")

    records: list[InferenceRecord] = []
    for row in rows:
        corrected_pred = row.get("corrected_pred_count")
        corrected_error = row.get("corrected_abs_error")
        records.append(
            InferenceRecord(
                image_id=str(row["image_id"]),
                gt_count=float(row["gt_count"]),
                pred_count=float(row["pred_count"]),
                abs_error=float(row["abs_error"]),
                corrected_pred_count=(
                    float(corrected_pred) if corrected_pred not in {None, ""} else None
                ),
                corrected_abs_error=(
                    float(corrected_error) if corrected_error not in {None, ""} else None
                ),
            )
        )
    return records


def metric_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must not be empty")
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "rmse": float(math.sqrt(np.mean(array * array))),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _corrected_errors(records: list[InferenceRecord]) -> np.ndarray:
    return np.asarray(
        [
            record.corrected_abs_error
            if record.corrected_abs_error is not None
            else record.abs_error
            for record in records
        ],
        dtype=np.float64,
    )


def _corrected_preds(records: list[InferenceRecord]) -> np.ndarray:
    return np.asarray(
        [
            record.corrected_pred_count
            if record.corrected_pred_count is not None
            else record.pred_count
            for record in records
        ],
        dtype=np.float64,
    )


def _setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_error_histogram(records: list[InferenceRecord], output_path: Path) -> None:
    raw = np.asarray([record.abs_error for record in records], dtype=np.float64)
    corrected = _corrected_errors(records)
    bins = np.linspace(0.0, max(float(raw.max()), float(corrected.max())) + 0.5, 32)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(raw, bins=bins, alpha=0.55, label="Raw abs error", color="#4C78A8")
    ax.hist(corrected, bins=bins, alpha=0.55, label="Calibrated abs error", color="#F58518")
    ax.axvline(raw.mean(), color="#4C78A8", linestyle="--", linewidth=1.5)
    ax.axvline(corrected.mean(), color="#F58518", linestyle="--", linewidth=1.5)
    ax.set_title("Absolute Error Distribution")
    ax.set_xlabel("Absolute error (count proxy)")
    ax.set_ylabel("Image count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pred_vs_gt(records: list[InferenceRecord], output_path: Path) -> None:
    gt = np.asarray([record.gt_count for record in records], dtype=np.float64)
    raw = np.asarray([record.pred_count for record in records], dtype=np.float64)
    corrected = _corrected_preds(records)
    limit = max(float(gt.max()), float(raw.max()), float(corrected.max())) + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, values, title, color in [
        (axes[0], raw, "Raw prediction", "#4C78A8"),
        (axes[1], corrected, "Calibrated prediction", "#F58518"),
    ]:
        ax.scatter(gt, values, s=14, alpha=0.55, color=color, edgecolors="none")
        ax.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("GT count proxy")
        ax.set_ylabel("Predicted count proxy")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
    fig.suptitle("Prediction vs GT")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def group_errors_by_gt(records: list[InferenceRecord]) -> list[tuple[str, int, float, float]]:
    bins = [(0, 8), (9, 12), (13, 16), (17, 20), (21, 10_000)]
    labels = ["0-8", "9-12", "13-16", "17-20", "21+"]
    rows: list[tuple[str, int, float, float]] = []
    corrected = _corrected_errors(records)
    for (lower, upper), label in zip(bins, labels, strict=True):
        indexes = [
            index
            for index, record in enumerate(records)
            if lower <= record.gt_count <= upper
        ]
        if not indexes:
            rows.append((label, 0, 0.0, 0.0))
            continue
        raw_values = np.asarray([records[index].abs_error for index in indexes], dtype=np.float64)
        corrected_values = corrected[indexes]
        rows.append((label, len(indexes), float(raw_values.mean()), float(corrected_values.mean())))
    return rows


def plot_gt_bucket_errors(records: list[InferenceRecord], output_path: Path) -> None:
    rows = group_errors_by_gt(records)
    labels = [row[0] for row in rows]
    counts = [row[1] for row in rows]
    raw = np.asarray([row[2] for row in rows], dtype=np.float64)
    corrected = np.asarray([row[3] for row in rows], dtype=np.float64)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, raw, width=width, label="Raw MAE", color="#4C78A8")
    ax.bar(x + width / 2, corrected, width=width, label="Calibrated MAE", color="#F58518")
    for index, count in enumerate(counts):
        ax.text(index, max(raw[index], corrected[index]) + 0.08, f"n={count}", ha="center")
    ax.set_title("MAE by GT Count Bucket")
    ax.set_xlabel("GT count proxy bucket")
    ax.set_ylabel("Mean absolute error")
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_error_boxplot(records: list[InferenceRecord], output_path: Path) -> None:
    raw = np.asarray([record.abs_error for record in records], dtype=np.float64)
    corrected = _corrected_errors(records)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([raw, corrected], labels=["Raw", "Calibrated"], showfliers=True)
    ax.set_title("Absolute Error Boxplot")
    ax.set_ylabel("Absolute error (count proxy)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_worst_cases(records: list[InferenceRecord], output_path: Path, worst_k: int) -> None:
    corrected_errors = _corrected_errors(records)
    order = np.argsort(corrected_errors)[::-1][: int(worst_k)]
    labels = [records[index].image_id for index in order][::-1]
    values = corrected_errors[order][::-1]

    fig_height = max(6, 0.26 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(labels, values, color="#E45756")
    ax.set_title(f"Worst {len(labels)} Calibrated Errors")
    ax.set_xlabel("Corrected absolute error")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_report(
    records: list[InferenceRecord],
    output_path: Path,
    title: str,
    figure_paths: list[Path],
) -> None:
    raw_summary = metric_summary(record.abs_error for record in records)
    corrected_summary = metric_summary(_corrected_errors(records))
    bucket_rows = group_errors_by_gt(records)
    worst = sorted(
        records,
        key=lambda record: (
            record.corrected_abs_error
            if record.corrected_abs_error is not None
            else record.abs_error
        ),
        reverse=True,
    )[:30]

    lines = [
        f"# {title}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Raw | Calibrated |",
        "|---|---:|---:|",
    ]
    for key in ["n", "mean", "rmse", "median", "p75", "p90", "p95", "max"]:
        lines.append(f"| {key} | {raw_summary[key]:.4f} | {corrected_summary[key]:.4f} |")

    lines.extend(
        [
            "",
            "## GT Bucket MAE",
            "",
            "| GT bucket | n | Raw MAE | Calibrated MAE |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, count, raw_mae, corrected_mae in bucket_rows:
        lines.append(f"| {label} | {count} | {raw_mae:.4f} | {corrected_mae:.4f} |")

    lines.extend(
        [
            "",
            "## Worst 30 Calibrated Errors",
            "",
            "| image_id | gt | pred | raw_err | corrected | cal_err |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in worst:
        corrected_pred = (
            record.corrected_pred_count
            if record.corrected_pred_count is not None
            else record.pred_count
        )
        corrected_error = (
            record.corrected_abs_error
            if record.corrected_abs_error is not None
            else record.abs_error
        )
        lines.append(
            f"| {record.image_id} | {record.gt_count:.0f} | {record.pred_count:.4f} | "
            f"{record.abs_error:.4f} | {corrected_pred:.4f} | {corrected_error:.4f} |"
        )

    lines.extend(["", "## Figures", ""])
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path.name}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_records(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot()

    figures = [
        args.output_dir / "01_error_histogram.png",
        args.output_dir / "02_prediction_vs_gt.png",
        args.output_dir / "03_gt_bucket_mae.png",
        args.output_dir / "04_error_boxplot.png",
        args.output_dir / "05_worst_calibrated_errors.png",
    ]
    plot_error_histogram(records, figures[0])
    plot_pred_vs_gt(records, figures[1])
    plot_gt_bucket_errors(records, figures[2])
    plot_error_boxplot(records, figures[3])
    plot_worst_cases(records, figures[4], worst_k=args.worst_k)
    write_report(
        records=records,
        output_path=args.output_dir / "inference_analysis_report.md",
        title=args.title,
        figure_paths=figures,
    )

    raw = metric_summary(record.abs_error for record in records)
    corrected = metric_summary(_corrected_errors(records))
    print(f"records={len(records)}")
    print(f"raw mae={raw['mean']:.4f} rmse={raw['rmse']:.4f} p90={raw['p90']:.4f}")
    print(
        f"calibrated mae={corrected['mean']:.4f} "
        f"rmse={corrected['rmse']:.4f} p90={corrected['p90']:.4f}"
    )
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
