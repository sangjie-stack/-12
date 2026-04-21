import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __package__:
    from .lego_size_detector import detect_circles
else:
    from detectors.lego_size_detector import detect_circles


@dataclass
class StackObject:
    index: int
    bbox: Tuple[int, int, int, int]
    circles: List[Tuple[float, float, float]]
    layers: int
    confidence: float


def build_color_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where((saturation > 45) & (value > 35), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def bounding_rect_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def cluster_circles(circles: Sequence[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
    if not circles:
        return []

    median_radius = float(np.median([r for _, _, r in circles]))
    distance_threshold = max(45.0, median_radius * 6.5)
    parent = list(range(len(circles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, _ = circles[i]
            x2, y2, _ = circles[j]
            if math.hypot(x1 - x2, y1 - y2) <= distance_threshold:
                union(i, j)

    groups = {}
    for index, circle in enumerate(circles):
        groups.setdefault(find(index), []).append(circle)

    clusters = [group for group in groups.values() if len(group) >= 3]
    clusters.sort(key=lambda group: min(circle[0] for circle in group))
    return clusters


def object_bbox_from_circles(image_shape: Sequence[int], circles: Sequence[Tuple[float, float, float]]) -> Tuple[int, int, int, int]:
    xs = np.array([x for x, _, _ in circles], dtype=np.float32)
    ys = np.array([y for _, y, _ in circles], dtype=np.float32)
    rs = np.array([r for _, _, r in circles], dtype=np.float32)
    pad_x = float(np.max(rs) * 2.5)
    pad_top = float(np.max(rs) * 1.3)
    pad_bottom = float(np.max(rs) * 15.0)

    x1 = max(0, int(np.floor(xs.min() - pad_x)))
    y1 = max(0, int(np.floor(ys.min() - pad_top)))
    x2 = min(image_shape[1] - 1, int(np.ceil(xs.max() + pad_x)))
    y2 = min(image_shape[0] - 1, int(np.ceil(ys.max() + pad_bottom)))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def count_color_bands(crop: np.ndarray) -> Tuple[int, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    center_x = crop.shape[1] // 2
    half_width = max(8, crop.shape[1] // 12)
    x1 = max(0, center_x - half_width)
    x2 = min(crop.shape[1], center_x + half_width)

    row_hues: List[int] = []
    row_presence: List[bool] = []
    for row_index in range(crop.shape[0]):
        valid = (saturation[row_index, x1:x2] > 35) & (value[row_index, x1:x2] > 30)
        if int(np.count_nonzero(valid)) < max(4, half_width // 2):
            row_presence.append(False)
            row_hues.append(-1)
            continue
        row_presence.append(True)
        row_hues.append(int(np.median(hsv[row_index, x1:x2, 0][valid])))

    raw_bands: List[Tuple[int, int, int]] = []
    current_hue = None
    start = 0
    for row_index, (present, hue) in enumerate(zip(row_presence, row_hues)):
        if not present:
            continue
        if current_hue is None:
            current_hue = hue
            start = row_index
            continue
        hue_gap = min(abs(hue - current_hue), 180 - abs(hue - current_hue))
        if hue_gap > 12:
            raw_bands.append((start, row_index - 1, current_hue))
            current_hue = hue
            start = row_index
    if current_hue is not None:
        raw_bands.append((start, len(row_hues) - 1, current_hue))

    min_band_height = max(14, crop.shape[0] // 12)
    substantial = [band for band in raw_bands if band[1] - band[0] + 1 >= min_band_height]
    if not substantial:
        return 1, 0.35

    confidence = float(np.clip(0.55 + 0.12 * len(substantial), 0.0, 0.95))
    return len(substantial), confidence


def detect_multi_stack_objects(image: np.ndarray) -> List[StackObject]:
    mask = build_color_mask(image)
    bbox = bounding_rect_from_mask(mask)
    circles = detect_circles(image, bbox)
    circles = [circle for circle in circles if circle[1] < image.shape[0] * 0.92]
    clusters = cluster_circles(circles)

    objects: List[StackObject] = []
    for index, cluster in enumerate(clusters, start=1):
        bbox = object_bbox_from_circles(image.shape, cluster)
        x, y, w, h = bbox
        crop = image[y: y + h, x: x + w]
        layers, confidence = count_color_bands(crop)
        objects.append(
            StackObject(
                index=index,
                bbox=bbox,
                circles=list(cluster),
                layers=layers,
                confidence=confidence,
            )
        )
    return objects


def draw_result(image: np.ndarray, objects: Sequence[StackObject]) -> np.ndarray:
    output = image.copy()
    palette = [(0, 220, 255), (0, 220, 0), (255, 180, 0), (255, 0, 180)]

    for object_result in objects:
        color = palette[(object_result.index - 1) % len(palette)]
        x, y, w, h = object_result.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        for center_x, center_y, radius in object_result.circles:
            cv2.circle(output, (int(round(center_x)), int(round(center_y))), int(round(radius)), color, 2)
            cv2.circle(output, (int(round(center_x)), int(round(center_y))), 2, color, -1)

        label = f"Object {object_result.index}: 1 x 1 x {object_result.layers} conf={object_result.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect multiple LEGO stack objects as 1 x 1 x layer-count.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional annotated output image path. Defaults to <input>_multi_stack.png.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return 1

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Could not read image: {args.image}")
        return 1

    objects = detect_multi_stack_objects(image)
    if not objects:
        print("No stack objects detected.")
        return 1

    for object_result in objects:
        print(f"Object {object_result.index}: 1 x 1 x {object_result.layers}")
        print(f"Confidence: {object_result.confidence:.2f}")

    output_path = args.output or args.image.with_name(f"{args.image.stem}_multi_stack.png")
    annotated = draw_result(image, objects)
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
