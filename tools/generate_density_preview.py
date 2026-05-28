from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.density_map import bbox_to_points, make_density_map
from src.utils.visualize import (
    density_to_heatmap,
    draw_annotation,
    draw_count_label,
    overlay_heatmap,
)
from src.utils.voc_parser import parse_voc_xml


SPLITS = {
    "train": Path("ImageSets/Main/train.txt"),
    "val": Path("ImageSets/Main/val.txt"),
    "test": Path("ImageSets/Main/test.txt"),
}


def read_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_items(data_root: Path, num_samples: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    split_ids = {
        split_name: read_split_ids(data_root / split_path)
        for split_name, split_path in SPLITS.items()
    }
    per_split = max(1, num_samples // len(split_ids))
    selected: list[tuple[str, str]] = []

    for split_name, ids in split_ids.items():
        sample_count = min(per_split, len(ids))
        selected.extend((split_name, image_id) for image_id in rng.sample(ids, sample_count))

    remaining = num_samples - len(selected)
    if remaining > 0:
        all_items = [
            (split_name, image_id)
            for split_name, ids in split_ids.items()
            for image_id in ids
        ]
        already = set(selected)
        candidates = [item for item in all_items if item not in already]
        selected.extend(rng.sample(candidates, min(remaining, len(candidates))))

    return selected


def write_preview(
    data_root: Path,
    output_dir: Path,
    split_name: str,
    image_id: str,
    sigma: float,
    downsample: int,
) -> dict[str, object]:
    annotation = parse_voc_xml(data_root / "Annotations" / f"{image_id}.xml")
    image_path = data_root / "Images" / annotation.filename
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    points = bbox_to_points(annotation.objects)
    density = make_density_map(
        points,
        height=annotation.height,
        width=annotation.width,
        sigma=sigma,
        downsample=downsample,
    )
    density_sum = float(density.sum())
    gt_count = len(points)

    item_dir = output_dir / split_name / image_id
    item_dir.mkdir(parents=True, exist_ok=True)

    original_path = item_dir / "original.jpg"
    annotation_path = item_dir / "annotation.jpg"
    heatmap_path = item_dir / "density_heatmap.jpg"
    overlay_path = item_dir / "density_overlay.jpg"

    if not cv2.imwrite(str(original_path), image):
        raise RuntimeError(f"Failed to write preview: {original_path}")
    annotation_image = draw_annotation(image, annotation)
    heatmap = density_to_heatmap(density, output_size=image.shape[:2])
    overlay = draw_count_label(
        overlay_heatmap(image, heatmap, alpha=0.4),
        annotation.image_id,
        gt_count=gt_count,
        density_sum=density_sum,
    )

    for path, rendered in (
        (annotation_path, annotation_image),
        (heatmap_path, heatmap),
        (overlay_path, overlay),
    ):
        if not cv2.imwrite(str(path), rendered):
            raise RuntimeError(f"Failed to write preview: {path}")

    return {
        "split": split_name,
        "image_id": image_id,
        "gt_count": gt_count,
        "density_sum": f"{density_sum:.6f}",
        "abs_error": f"{abs(density_sum - gt_count):.6f}",
        "sigma": sigma,
        "downsample": downsample,
        "preview_dir": item_dir.as_posix(),
    }


def check_integral(
    data_root: Path,
    split_name: str,
    image_id: str,
    sigma: float,
    downsample: int,
    preview_dir: Path | None = None,
) -> dict[str, object]:
    annotation = parse_voc_xml(data_root / "Annotations" / f"{image_id}.xml")
    points = bbox_to_points(annotation.objects)
    density = make_density_map(
        points,
        height=annotation.height,
        width=annotation.width,
        sigma=sigma,
        downsample=downsample,
    )
    density_sum = float(density.sum())
    gt_count = len(points)
    return {
        "split": split_name,
        "image_id": image_id,
        "gt_count": gt_count,
        "density_sum": f"{density_sum:.6f}",
        "abs_error": f"{abs(density_sum - gt_count):.6f}",
        "sigma": sigma,
        "downsample": downsample,
        "preview_dir": preview_dir.as_posix() if preview_dir else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FDU density map previews.")
    parser.add_argument("--data-root", type=Path, default=Path("FDU_HairFollicleDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/density_previews"))
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--check-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--downsample", type=int, default=8)
    args = parser.parse_args()

    preview_items = sample_items(args.data_root, args.num_samples, args.seed)
    check_items = sample_items(args.data_root, max(args.check_samples, args.num_samples), args.seed)
    rows: list[dict[str, object]] = []

    preview_set = set(preview_items)
    for split_name, image_id in preview_items:
        write_preview(
            args.data_root,
            args.output_dir,
            split_name,
            image_id,
            sigma=args.sigma,
            downsample=args.downsample,
        )

    for split_name, image_id in check_items:
        preview_dir = (
            args.output_dir / split_name / image_id
            if (split_name, image_id) in preview_set
            else None
        )
        row = check_integral(
            args.data_root,
            split_name,
            image_id,
            sigma=args.sigma,
            downsample=args.downsample,
            preview_dir=preview_dir,
        )
        rows.append(row)

    report_path = args.output_dir / "density_integral_check.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "image_id",
                "gt_count",
                "density_sum",
                "abs_error",
                "sigma",
                "downsample",
                "preview_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    max_error = max(float(row["abs_error"]) for row in rows) if rows else 0.0
    print(
        f"Wrote {len(preview_items)} density previews to {args.output_dir}; "
        f"checked {len(rows)} maps, max abs integral error={max_error:.6f}; "
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
