from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.voc_parser import parse_voc_xml, raw_bbox_has_reversed_order


SPLITS = {
    "train": Path("ImageSets/Main/train.txt"),
    "val": Path("ImageSets/Main/val.txt"),
    "test": Path("ImageSets/Main/test.txt"),
}


def read_split_ids(data_root: Path, split_path: Path) -> list[str]:
    path = data_root / split_path
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def analyze(data_root: Path) -> dict[str, object]:
    annotations_dir = data_root / "Annotations"
    images_dir = data_root / "Images"
    xml_paths = sorted(annotations_dir.glob("*.xml"))
    image_paths = sorted(images_dir.glob("*"))

    class_counts: Counter[str] = Counter()
    raw_class_counts: Counter[str] = Counter()
    image_counts: list[int] = []
    raw_image_counts: list[int] = []
    raw_reversed_count = 0
    invalid_count = 0
    width_heights: Counter[tuple[int, int]] = Counter()

    for xml_path in xml_paths:
        annotation = parse_voc_xml(xml_path)
        image_counts.append(len(annotation.objects))
        raw_image_counts.append(len(annotation.objects) + len(annotation.invalid_objects))
        width_heights[(annotation.width, annotation.height)] += 1
        invalid_count += len(annotation.invalid_objects)
        for obj in annotation.objects:
            class_counts[obj.class_name] += 1
            raw_class_counts[obj.class_name] += 1
            if raw_bbox_has_reversed_order(obj.raw_bbox):
                raw_reversed_count += 1
        for invalid in annotation.invalid_objects:
            raw_class_counts[invalid.class_name] += 1
            if invalid.raw_bbox and raw_bbox_has_reversed_order(invalid.raw_bbox):
                raw_reversed_count += 1

    split_ids = {name: read_split_ids(data_root, split) for name, split in SPLITS.items()}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    overlaps = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    split_union_count = len(set().union(*split_sets.values())) if split_sets else 0

    return {
        "xml_count": len(xml_paths),
        "image_count": len([p for p in image_paths if p.is_file()]),
        "class_counts": class_counts,
        "raw_class_counts": raw_class_counts,
        "image_counts": image_counts,
        "raw_image_counts": raw_image_counts,
        "raw_reversed_count": raw_reversed_count,
        "invalid_count": invalid_count,
        "width_heights": width_heights,
        "split_ids": split_ids,
        "overlaps": overlaps,
        "split_union_count": split_union_count,
    }


def build_report(data_root: Path, result: dict[str, object]) -> str:
    image_counts = result["image_counts"]
    assert isinstance(image_counts, list)
    class_counts = result["class_counts"]
    raw_class_counts = result["raw_class_counts"]
    width_heights = result["width_heights"]
    split_ids = result["split_ids"]
    overlaps = result["overlaps"]
    assert isinstance(class_counts, Counter)
    assert isinstance(raw_class_counts, Counter)
    assert isinstance(width_heights, Counter)
    assert isinstance(split_ids, dict)
    assert isinstance(overlaps, dict)

    lines = [
        "# FDU 数据统计报告",
        "",
        "本报告由 `tools/analyze_fdu.py` 生成。当前项目使用 FDU VOC bbox 中心点作为毛囊/标注目标计数 proxy。",
        "",
        "## 基础统计",
        "",
        markdown_table(
            ["项目", "数量"],
            [
                ["数据集路径", data_root.as_posix()],
                ["图片数量", result["image_count"]],
                ["XML 标注数量", result["xml_count"]],
                ["原始标注目标数", sum(result["raw_image_counts"])],
                ["有效标注目标数", sum(image_counts)],
                ["有效目标平均每图数量", f"{statistics.mean(image_counts):.2f}" if image_counts else "0.00"],
                ["有效目标每图最小值", min(image_counts) if image_counts else 0],
                ["有效目标每图最大值", max(image_counts) if image_counts else 0],
                ["原始 bbox 坐标顺序异常数", result["raw_reversed_count"]],
                ["修正裁剪后仍无效 bbox 数", result["invalid_count"]],
            ],
        ),
        "",
        "## 图片分辨率",
        "",
        markdown_table(
            ["宽", "高", "数量"],
            [[width, height, count] for (width, height), count in sorted(width_heights.items())],
        ),
        "",
        "## 原始类别分布",
        "",
        markdown_table(
            ["类别", "数量"],
            [[name, count] for name, count in raw_class_counts.most_common()],
        ),
        "",
        "## 有效类别分布",
        "",
        markdown_table(
            ["类别", "数量"],
            [[name, count] for name, count in class_counts.most_common()],
        ),
        "",
        "## 数据划分",
        "",
        markdown_table(
            ["划分", "数量"],
            [[name, len(ids)] for name, ids in split_ids.items()]
            + [["合并去重后数量", result["split_union_count"]]],
        ),
        "",
        "## 划分交叉检查",
        "",
        markdown_table(
            ["检查项", "交叉数量"],
            [
                ["train ∩ val", overlaps["train_val"]],
                ["train ∩ test", overlaps["train_test"]],
                ["val ∩ test", overlaps["val_test"]],
            ],
        ),
        "",
        "## 数据质量结论",
        "",
        "- VOC XML 可解析，bbox 已按 min/max 规则修正坐标顺序。",
        "- 修正并裁剪后仍无效的 bbox 已在解析阶段过滤。",
        "- 当前输出不代表真实业务 `根/cm^2`，只作为基于 FDU bbox 中心点的密度估计 baseline 数据基础。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FDU VOC annotations.")
    parser.add_argument("--data-root", type=Path, default=Path("FDU_HairFollicleDataset"))
    parser.add_argument("--output", type=Path, default=Path("docs/data_report.md"))
    args = parser.parse_args()

    result = analyze(args.data_root)
    report = build_report(args.data_root, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
