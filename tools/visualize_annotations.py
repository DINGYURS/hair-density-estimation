from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.visualize import save_annotation_preview
from src.utils.voc_parser import parse_voc_xml


def read_split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FDU bbox annotation previews.")
    parser.add_argument("--data-root", type=Path, default=Path("FDU_HairFollicleDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/annotations"))
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_paths = {
        "train": args.data_root / "ImageSets/Main/train.txt",
        "val": args.data_root / "ImageSets/Main/val.txt",
        "test": args.data_root / "ImageSets/Main/test.txt",
    }
    rng = random.Random(args.seed)
    per_split = max(1, args.num_samples // len(split_paths))
    selected: list[tuple[str, str]] = []

    for split_name, split_path in split_paths.items():
        ids = read_split_ids(split_path)
        sample_count = min(per_split, len(ids))
        selected.extend((split_name, image_id) for image_id in rng.sample(ids, sample_count))

    remaining = args.num_samples - len(selected)
    if remaining > 0:
        all_items = [
            (split_name, image_id)
            for split_name, split_path in split_paths.items()
            for image_id in read_split_ids(split_path)
        ]
        already = set(selected)
        candidates = [item for item in all_items if item not in already]
        selected.extend(rng.sample(candidates, min(remaining, len(candidates))))

    for split_name, image_id in selected:
        xml_path = args.data_root / "Annotations" / f"{image_id}.xml"
        annotation = parse_voc_xml(xml_path)
        image_path = args.data_root / "Images" / annotation.filename
        output_path = args.output_dir / split_name / f"{image_id}.jpg"
        save_annotation_preview(image_path, annotation, output_path)

    print(f"Wrote {len(selected)} annotation previews to {args.output_dir}")


if __name__ == "__main__":
    main()
