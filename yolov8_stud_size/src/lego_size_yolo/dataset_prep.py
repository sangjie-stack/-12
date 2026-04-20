from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .geometry import box_to_quad, clip_quad_to_image, expand_quad, mask_to_quad, warp_from_quad


@dataclass
class PreparedCrop:
    source_image: str
    source_label: str
    output_image: str
    output_label: str
    split: str
    index: int
    class_id: int
    warp_width: int
    warp_height: int
    quad: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_label_line(line: str) -> Dict[str, Any]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line: {line}")

    class_id = int(float(parts[0]))
    coords = [float(value) for value in parts[1:]]
    if len(coords) == 4:
        return {"class_id": class_id, "type": "bbox", "values": coords}
    if len(coords) >= 6 and len(coords) % 2 == 0:
        return {"class_id": class_id, "type": "polygon", "values": coords}
    raise ValueError(f"Unsupported YOLO label format: {line}")


def load_yolo_labels(label_path: Path) -> List[Dict[str, Any]]:
    if not label_path.exists():
        return []

    objects: List[Dict[str, Any]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        objects.append(_parse_label_line(line))
    return objects


def _bbox_area_from_normalized(values: Sequence[float]) -> float:
    _, _, width, height = values
    return float(width) * float(height)


def _polygon_area_from_normalized(values: Sequence[float]) -> float:
    points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def choose_primary_object(objects: Sequence[Dict[str, Any]], class_id: int = 0) -> Dict[str, Any]:
    filtered = [obj for obj in objects if obj["class_id"] == class_id]
    if not filtered:
        raise ValueError(f"No label with class_id={class_id} found.")

    def score(item: Dict[str, Any]) -> float:
        if item["type"] == "bbox":
            return _bbox_area_from_normalized(item["values"])
        return _polygon_area_from_normalized(item["values"])

    return max(filtered, key=score)


def object_to_quad(obj: Dict[str, Any], image_shape: Sequence[int]) -> np.ndarray:
    height, width = image_shape[:2]
    if obj["type"] == "bbox":
        center_x, center_y, box_width, box_height = obj["values"]
        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height
        return box_to_quad([x1, y1, x2, y2])

    polygon = np.asarray(obj["values"], dtype=np.float32).reshape(-1, 2)
    polygon[:, 0] *= width
    polygon[:, 1] *= height
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask_to_quad(mask)


def build_crop_from_label(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    split: str,
    index: int = 0,
    target_class_id: int = 0,
    quad_scale: float = 1.04,
) -> PreparedCrop:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    objects = load_yolo_labels(label_path)
    selected = choose_primary_object(objects, class_id=target_class_id)
    quad = object_to_quad(selected, image.shape)
    quad = expand_quad(quad, scale=quad_scale)
    quad = clip_quad_to_image(quad, image.shape)
    warped, _ = warp_from_quad(image, quad)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), warped)
    output_label_path.touch(exist_ok=True)

    return PreparedCrop(
        source_image=str(image_path),
        source_label=str(label_path),
        output_image=str(output_image_path),
        output_label=str(output_label_path),
        split=split,
        index=index,
        class_id=int(selected["class_id"]),
        warp_width=int(warped.shape[1]),
        warp_height=int(warped.shape[0]),
        quad=quad.astype(float).tolist(),
    )


def save_manifest(items: Sequence[PreparedCrop], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = [item.to_dict() for item in items]
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
