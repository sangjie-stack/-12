from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from lego_multi_stack_detector import build_color_mask, count_color_bands


@dataclass
class GeneratedObject:
    index: int
    bbox: Tuple[int, int, int, int]
    circles: List[Tuple[float, float, float]]
    dims: Tuple[int, int]
    height: int
    confidence: float


def detect_generated_studs(crop: np.ndarray) -> List[Tuple[float, float, float]]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    stud_mask = np.where((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 180), 255, 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(stud_mask, connectivity=8)

    studs: List[Tuple[float, float, float]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 80 or area > 450:
            continue
        cx, cy = centroids[label]
        radius = max(w, h) / 2.0
        studs.append((float(cx), float(cy), float(radius)))
    return studs


def cluster_values(values: Sequence[float], tolerance: float) -> List[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []

    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(sum(cluster) / len(cluster)) for cluster in clusters]


def infer_generated_dims(studs: Sequence[Tuple[float, float, float]]) -> Tuple[Tuple[int, int], float]:
    if not studs:
        return (0, 0), 0.0
    if len(studs) == 1:
        return (1, 1), 0.95

    row_centers = cluster_values([y for _, y, _ in studs], tolerance=6.0)
    if not row_centers:
        return (0, 0), 0.0

    adjusted_xs = []
    for x, y, _ in studs:
        row_index = min(range(len(row_centers)), key=lambda index: abs(y - row_centers[index]))
        adjusted_xs.append(float(x + 18.0 * row_index))

    col_centers = cluster_values(adjusted_xs, tolerance=14.0)
    length = max(1, len(col_centers))
    width = max(1, len(row_centers))
    expected = max(1, length * width)
    coverage = min(1.0, len(studs) / float(expected))
    confidence = float(np.clip(0.65 + 0.30 * coverage, 0.0, 0.98))
    return (length, width), confidence


def detect_generated_objects(image: np.ndarray) -> List[GeneratedObject]:
    mask = build_color_mask(image)
    objects: List[GeneratedObject] = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 1000:
            continue
        candidates.append((x, y, w, h, area))

    candidates.sort(key=lambda item: item[0])
    for index, (x, y, w, h, _) in enumerate(candidates, start=1):
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        crop = image[y1:y2, x1:x2]

        local_circles = detect_generated_studs(crop)
        dims, size_confidence = infer_generated_dims(local_circles)
        raw_height, height_confidence = count_color_bands(crop)
        height = max(1, raw_height - 1)
        confidence = float(np.clip(0.55 * size_confidence + 0.45 * height_confidence, 0.0, 1.0))

        objects.append(
            GeneratedObject(
                index=index,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                circles=[(float(cx + x1), float(cy + y1), float(r)) for cx, cy, r in local_circles],
                dims=dims,
                height=height,
                confidence=confidence,
            )
        )

    return objects


def draw_generated_result(image: np.ndarray, objects: Sequence[GeneratedObject]) -> np.ndarray:
    output = image.copy()
    palette = [(0, 220, 255), (0, 220, 0), (255, 180, 0), (255, 0, 180)]

    for object_result in objects:
        color = palette[(object_result.index - 1) % len(palette)]
        x, y, w, h = object_result.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        for center_x, center_y, radius in object_result.circles:
            cv2.circle(output, (int(round(center_x)), int(round(center_y))), int(round(radius)), color, 2)
            cv2.circle(output, (int(round(center_x)), int(round(center_y))), 2, color, -1)

        label = f"Object {object_result.index}: {object_result.dims[0]} x {object_result.dims[1]} x {object_result.height} conf={object_result.confidence:.2f}"
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
