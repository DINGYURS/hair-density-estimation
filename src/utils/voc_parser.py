from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class VocObject:
    class_name: str
    bbox: tuple[float, float, float, float]
    raw_bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class InvalidVocObject:
    class_name: str
    raw_bbox: tuple[float, float, float, float] | None
    reason: str


@dataclass(frozen=True)
class VocAnnotation:
    image_id: str
    filename: str
    width: int
    height: int
    objects: list[VocObject]
    invalid_objects: list[InvalidVocObject]


def _required_text(parent: ET.Element, tag: str) -> str:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Missing required XML tag: {tag}")
    return node.text.strip()


def _find_class_name(object_node: ET.Element) -> str:
    for tag in ("name", "class"):
        node = object_node.find(tag)
        if node is not None and node.text:
            return node.text.strip()
    return "unknown"


def _read_bbox(object_node: ET.Element) -> tuple[float, float, float, float]:
    box = object_node.find("bndbox")
    if box is None:
        raise ValueError("Missing bndbox")
    return (
        float(_required_text(box, "xmin")),
        float(_required_text(box, "ymin")),
        float(_required_text(box, "xmax")),
        float(_required_text(box, "ymax")),
    )


def parse_voc_xml(xml_path: str | Path) -> VocAnnotation:
    """Parse one VOC XML file and normalize invalid bbox coordinate order."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    filename = _required_text(root, "filename")
    image_id = Path(filename).stem or xml_path.stem
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size tag in {xml_path}")
    width = int(float(_required_text(size, "width")))
    height = int(float(_required_text(size, "height")))

    objects: list[VocObject] = []
    invalid_objects: list[InvalidVocObject] = []

    for object_node in root.findall("object"):
        class_name = _find_class_name(object_node)
        try:
            raw_bbox = _read_bbox(object_node)
        except ValueError as exc:
            invalid_objects.append(
                InvalidVocObject(class_name=class_name, raw_bbox=None, reason=str(exc))
            )
            continue

        xmin, ymin, xmax, ymax = raw_bbox
        x1 = max(0.0, min(xmin, xmax))
        y1 = max(0.0, min(ymin, ymax))
        x2 = min(float(width), max(xmin, xmax))
        y2 = min(float(height), max(ymin, ymax))

        if x2 <= x1 or y2 <= y1:
            invalid_objects.append(
                InvalidVocObject(
                    class_name=class_name,
                    raw_bbox=raw_bbox,
                    reason="non_positive_area_after_clip",
                )
            )
            continue

        objects.append(
            VocObject(
                class_name=class_name,
                bbox=(x1, y1, x2, y2),
                raw_bbox=raw_bbox,
            )
        )

    return VocAnnotation(
        image_id=image_id,
        filename=filename,
        width=width,
        height=height,
        objects=objects,
        invalid_objects=invalid_objects,
    )


def raw_bbox_has_reversed_order(raw_bbox: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = raw_bbox
    return xmin > xmax or ymin > ymax

