from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def mask_to_quad(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask.")

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    quad = cv2.boxPoints(rect)
    return order_quad_points(quad)


def box_to_quad(box_xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    quad = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    return order_quad_points(quad)


def warp_from_quad(image: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    quad = order_quad_points(quad)
    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])

    width = max(32, int(round(max(width_top, width_bottom))))
    height = max(32, int(round(max(height_left, height_right))))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def expand_quad(quad: np.ndarray, scale: float = 1.0) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32)
    center = np.mean(quad, axis=0, keepdims=True)
    expanded = (quad - center) * float(scale) + center
    return expanded.astype(np.float32)


def clip_quad_to_image(quad: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    height, width = image_shape[:2]
    clipped = np.asarray(quad, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def cluster_1d(values: Iterable[float], tolerance_ratio: float = 0.45) -> List[float]:
    values = sorted(float(value) for value in values)
    if not values:
        return []
    if len(values) == 1:
        return [values[0]]

    diffs = np.diff(np.array(values, dtype=np.float32))
    positive_diffs = diffs[diffs > 1e-6]
    if len(positive_diffs) == 0:
        return [float(np.mean(values))]

    reference_gap = float(np.median(positive_diffs))
    tolerance = max(4.0, reference_gap * tolerance_ratio)

    clusters: List[List[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def infer_grid_from_centers(points: Sequence[Sequence[float]]) -> Tuple[Tuple[int, int], float]:
    if not points:
        return (0, 0), 0.0

    points_array = np.asarray(points, dtype=np.float32)
    x_clusters = cluster_1d(points_array[:, 0])
    y_clusters = cluster_1d(points_array[:, 1])

    cols = len(x_clusters)
    rows = len(y_clusters)
    if rows == 0 or cols == 0:
        return (0, 0), 0.0

    expected = rows * cols
    observed = len(points_array)
    coverage = min(1.0, observed / float(expected)) if expected else 0.0

    x_steps = np.diff(np.array(x_clusters, dtype=np.float32)) if cols > 1 else np.array([1.0], dtype=np.float32)
    y_steps = np.diff(np.array(y_clusters, dtype=np.float32)) if rows > 1 else np.array([1.0], dtype=np.float32)
    x_reg = 1.0 if len(x_steps) <= 1 else max(0.0, 1.0 - float(np.std(x_steps) / (np.mean(x_steps) + 1e-6)))
    y_reg = 1.0 if len(y_steps) <= 1 else max(0.0, 1.0 - float(np.std(y_steps) / (np.mean(y_steps) + 1e-6)))
    confidence = float(np.clip(0.55 * coverage + 0.225 * x_reg + 0.225 * y_reg, 0.0, 1.0))

    canonical = tuple(sorted((rows, cols)))
    return canonical, confidence
