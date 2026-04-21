import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __package__:
    from .lego_height_detector import detect_lego_height
else:
    from detectors.lego_height_detector import detect_lego_height


@dataclass
class CircleFeature:
    circle: Tuple[float, float, float]
    mask_ratio: float
    polarity: float
    rel_y: float
    center_bias: float
    radius_bias: float
    score: float


@dataclass
class DetectionResult:
    dims: Tuple[int, int]
    size_confidence: float
    height: int
    height_confidence: float
    circles: List[Tuple[float, float, float]]
    top_circles: List[Tuple[float, float, float]]
    bbox: Tuple[int, int, int, int]
    top_face_quad: Optional[List[Tuple[float, float]]]
    height_lines: List[int]
    message: str


def refine_compact_height(
    dims: Tuple[int, int],
    top_circles: Sequence[Tuple[float, float, float]],
    raw_height: int,
    raw_confidence: float,
) -> Tuple[int, float]:
    if dims == (2, 2) and len(top_circles) == 4 and raw_height in (3, 4):
        return 2, float(max(0.88, raw_confidence * 0.98))
    return raw_height, raw_confidence


def shape_3d_string(result: DetectionResult) -> Optional[str]:
    if result.dims == (0, 0) or result.height <= 0:
        return None
    return f"{result.dims[0]} x {result.dims[1]} x {result.height}"


def build_foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colorful = hsv[:, :, 1] > 28
    darker = gray < 242
    mask = np.where(colorful | darker, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_contour_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, mask.shape[1], mask.shape[0]
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def resize_for_detection(image: np.ndarray, max_side: int = 1100) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    current_max = max(height, width)
    scale = max_side / float(current_max)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized, scale


def circle_quality(gray: np.ndarray, edges: np.ndarray, circle: Tuple[float, float, float]) -> float:
    x, y, r = circle
    x_i = int(round(x))
    y_i = int(round(y))
    r_i = max(4, int(round(r)))

    if (
        x_i - r_i - 3 < 0
        or y_i - r_i - 3 < 0
        or x_i + r_i + 3 >= gray.shape[1]
        or y_i + r_i + 3 >= gray.shape[0]
    ):
        return 0.0

    sample_points = 48
    hits = 0
    for angle in np.linspace(0.0, 2.0 * np.pi, sample_points, endpoint=False):
        px = int(round(x_i + math.cos(angle) * r_i))
        py = int(round(y_i + math.sin(angle) * r_i))
        ring_patch = edges[max(0, py - 2): py + 3, max(0, px - 2): px + 3]
        if ring_patch.size > 0 and np.max(ring_patch) > 0:
            hits += 1
    edge_ratio = hits / float(sample_points)

    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    center_mask = distance <= r * 0.45
    annulus_mask = (distance >= r * 0.70) & (distance <= r * 1.05)
    if not np.any(center_mask) or not np.any(annulus_mask):
        return 0.0

    center_value = float(np.mean(gray[center_mask]))
    ring_value = float(np.mean(gray[annulus_mask]))
    contrast = min(1.0, abs(center_value - ring_value) / 40.0)
    return 0.75 * edge_ratio + 0.25 * contrast


def detect_circles(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[float, float, float]]:
    resized, scale = resize_for_detection(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blurred, 60, 140)

    _, _, bw, bh = bbox
    bbox_min = max(20, min(bw, bh))
    min_radius = max(6, int(bbox_min * scale * 0.035))
    max_radius = max(min_radius + 4, int(bbox_min * scale * 0.10))
    min_dist = max(12, int(min_radius * 1.4))

    scored_candidates: List[Tuple[float, float, float, float]] = []
    for param2 in (20, 24, 28):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=90,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue
        for x, y, r in circles[0]:
            score = circle_quality(gray, edges, (float(x), float(y), float(r)))
            if score < 0.38:
                continue
            scored_candidates.append((x / scale, y / scale, r / scale, score))

    return dedupe_circles(scored_candidates)


def dedupe_circles(circles: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float]]:
    kept_scored: List[Tuple[float, float, float, float]] = []
    for x, y, r, score in sorted(circles, key=lambda item: item[3], reverse=True):
        duplicate = False
        for kept_x, kept_y, kept_r, kept_score in kept_scored:
            center_distance = math.hypot(x - kept_x, y - kept_y)
            radius_distance = abs(r - kept_r)
            if center_distance <= max(r, kept_r) * 0.55 and radius_distance <= max(r, kept_r) * 0.35:
                duplicate = True
                if score > kept_score:
                    kept_scored.remove((kept_x, kept_y, kept_r, kept_score))
                break
        if not duplicate:
            kept_scored.append((x, y, r, score))

    kept = [(x, y, r) for x, y, r, _ in kept_scored]
    return sorted(kept, key=lambda item: (item[1], item[0]))


def circle_mask_ratio(mask: np.ndarray, circle: Tuple[float, float, float]) -> float:
    x, y, r = circle
    sample = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(sample, (int(round(x)), int(round(y))), int(round(r * 0.85)), 255, -1)
    overlap = cv2.bitwise_and(mask, sample)
    denom = max(1, cv2.countNonZero(sample))
    return cv2.countNonZero(overlap) / float(denom)


def circle_region_values(gray: np.ndarray, circle: Tuple[float, float, float]) -> Tuple[float, float]:
    x, y, r = circle
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    center_mask = distance <= r * 0.45
    annulus_mask = (distance >= r * 0.70) & (distance <= r * 1.05)
    if not np.any(center_mask) or not np.any(annulus_mask):
        return 0.0, 0.0
    center_value = float(np.mean(gray[center_mask]))
    ring_value = float(np.mean(gray[annulus_mask]))
    return center_value, ring_value


def build_circle_features(
    circles: Sequence[Tuple[float, float, float]],
    gray: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> List[CircleFeature]:
    bx, by, bw, bh = bbox
    if not circles:
        return []

    median_radius = float(np.median([r for _, _, r in circles]))
    center_x = bx + bw * 0.5
    features: List[CircleFeature] = []
    for circle in circles:
        x, y, r = circle
        rel_y = (y - by) / max(1.0, float(bh))
        if rel_y < -0.05 or rel_y > 0.62:
            continue

        mask_ratio = circle_mask_ratio(mask, circle)
        if mask_ratio < 0.45:
            continue

        center_value, ring_value = circle_region_values(gray, circle)
        polarity = center_value - ring_value
        center_bias = max(0.0, 1.0 - abs(x - center_x) / max(1.0, bw * 0.55))
        radius_bias = max(0.0, 1.0 - abs(r - median_radius) / max(1.0, median_radius))
        score = (
            1.20 * math.tanh(polarity / 18.0)
            + 0.90 * (1.0 - rel_y)
            + 0.35 * center_bias
            + 0.25 * radius_bias
            + 0.30 * mask_ratio
        )
        features.append(
            CircleFeature(
                circle=circle,
                mask_ratio=mask_ratio,
                polarity=polarity,
                rel_y=rel_y,
                center_bias=center_bias,
                radius_bias=radius_bias,
                score=score,
            )
        )
    return features


def cluster_axis(values: np.ndarray) -> List[float]:
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [float(values[0])]

    ordered = np.sort(values.astype(np.float32))
    diffs = np.diff(ordered)
    positive = diffs[diffs > 1.0]
    if len(positive) == 0:
        return [float(np.mean(ordered))]

    spacing = float(np.median(positive))
    tolerance = max(4.0, spacing * 0.45)

    clusters: List[List[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return [float(np.mean(cluster)) for cluster in clusters]


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def infer_small_layout(circles: Sequence[Tuple[float, float, float]]) -> Optional[Tuple[Tuple[int, int], float]]:
    points = np.array([[x, y] for x, y, _ in circles], dtype=np.float32)
    if len(points) < 2:
        return None

    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    major = float(singular_values[0]) if len(singular_values) >= 1 else 0.0
    minor = float(singular_values[1]) if len(singular_values) >= 2 else 0.0
    spread_ratio = minor / max(major, 1e-6)

    if len(points) == 2:
        return (1, 2), float(np.clip(0.65 + 0.20 * (1.0 - spread_ratio), 0.0, 1.0))

    if len(points) == 3:
        if spread_ratio < 0.15:
            return (1, 3), 0.70
        return (2, 2), 0.55

    if len(points) == 4:
        hull = cv2.convexHull(points).reshape(-1, 2)
        area = polygon_area(hull)
        bbox_area = max(1.0, float((points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())))
        area_ratio = area / bbox_area
        if area_ratio >= 0.34:
            confidence = float(np.clip(0.70 + 0.24 * area_ratio, 0.0, 1.0))
            return (2, 2), confidence
        if spread_ratio < 0.18:
            return (1, 4), 0.78
        confidence = float(np.clip(0.72 + 0.20 * area_ratio, 0.0, 1.0))
        return (2, 2), confidence

    if len(points) == 5 and spread_ratio < 0.16:
        return (1, 5), 0.72

    return None


def infer_dims_from_centers(circles: Sequence[Tuple[float, float, float]]) -> Tuple[Tuple[int, int], float]:
    centers = np.array([[x, y] for x, y, _ in circles], dtype=np.float32)
    if len(centers) == 0:
        return (0, 0), 0.0
    if len(centers) == 1:
        return (1, 1), 0.2

    small_layout = infer_small_layout(circles)
    if small_layout is not None:
        return small_layout

    centered = centers - centers.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]
    projections = centered @ axes.T

    axis_a = cluster_axis(projections[:, 0])
    axis_b = cluster_axis(projections[:, 1])

    count_a = max(1, len(axis_a))
    count_b = max(1, len(axis_b))
    dims = tuple(sorted((count_a, count_b)))

    expected = max(1, count_a * count_b)
    coverage = min(1.0, len(circles) / float(expected))

    regularity_scores = []
    for axis_values in (axis_a, axis_b):
        if len(axis_values) <= 2:
            regularity_scores.append(1.0)
            continue
        steps = np.diff(np.array(axis_values, dtype=np.float32))
        mean_step = float(np.mean(steps))
        if mean_step <= 1e-5:
            regularity_scores.append(0.0)
            continue
        deviation = float(np.std(steps) / mean_step)
        regularity_scores.append(max(0.0, 1.0 - deviation))

    confidence = float(np.clip(0.55 * coverage + 0.45 * np.mean(regularity_scores), 0.0, 1.0))
    return dims, confidence


def evaluate_circle_subset(features: Sequence[CircleFeature]) -> Tuple[float, Tuple[int, int], float]:
    circles = [feature.circle for feature in features]
    dims, confidence = infer_dims_from_centers(circles)
    if dims == (0, 0):
        return -1.0, dims, confidence

    expected = max(1, dims[0] * dims[1])
    coverage = min(1.0, len(circles) / float(expected))
    overflow = max(0.0, len(circles) - expected)
    polarity_bonus = float(np.mean([max(0.0, feature.polarity) for feature in features]))
    score = confidence + 0.10 * coverage + 0.05 * min(1.0, polarity_bonus / 18.0) - 0.06 * overflow
    if expected > 16:
        score -= 0.15
    return score, dims, confidence


def select_top_face_circles(
    circles: Sequence[Tuple[float, float, float]],
    gray: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> List[Tuple[float, float, float]]:
    features = build_circle_features(circles, gray, mask, bbox)
    if len(features) < 4:
        return []

    best_subset: List[CircleFeature] = []
    best_score = -1.0

    for rel_y_limit in (0.40, 0.46, 0.52):
        for polarity_floor in (2.0, 0.0, -4.0):
            filtered = [feature for feature in features if feature.rel_y <= rel_y_limit and feature.polarity >= polarity_floor]
            filtered.sort(key=lambda feature: feature.score, reverse=True)
            for keep_count in range(4, min(14, len(filtered)) + 1):
                subset = filtered[:keep_count]
                subset_score, _, _ = evaluate_circle_subset(subset)
                if subset_score > best_score:
                    best_score = subset_score
                    best_subset = subset

    features.sort(key=lambda feature: feature.score, reverse=True)
    for keep_count in range(4, min(14, len(features)) + 1):
        subset = features[:keep_count]
        subset_score, _, _ = evaluate_circle_subset(subset)
        if subset_score > best_score:
            best_score = subset_score
            best_subset = subset

    if not best_subset:
        return []

    best_subset.sort(key=lambda feature: (feature.circle[1], feature.circle[0]))
    subset_score, dims, _ = evaluate_circle_subset(best_subset)
    expected = dims[0] * dims[1]
    if expected > 0 and len(best_subset) > expected:
        best_subset = sorted(best_subset, key=lambda feature: feature.score, reverse=True)[:expected]
        best_subset.sort(key=lambda feature: (feature.circle[1], feature.circle[0]))
    return [feature.circle for feature in best_subset]


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def axis_span_padding(axis_values: List[float], fallback: float) -> float:
    if len(axis_values) <= 1:
        return fallback
    steps = np.diff(np.array(axis_values, dtype=np.float32))
    if len(steps) == 0:
        return fallback
    return max(fallback, float(np.median(steps)) * 0.75)


def estimate_top_face_quad(circles: Sequence[Tuple[float, float, float]]) -> Optional[np.ndarray]:
    if len(circles) < 4:
        return None

    points = np.array([[x, y] for x, y, _ in circles], dtype=np.float32)
    radii = np.array([r for _, _, r in circles], dtype=np.float32)
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]
    projections = centered @ axes.T

    axis_u = cluster_axis(projections[:, 0])
    axis_v = cluster_axis(projections[:, 1])
    if not axis_u or not axis_v:
        return None

    fallback = float(np.median(radii)) * 1.8
    pad_u = axis_span_padding(axis_u, fallback)
    pad_v = axis_span_padding(axis_v, fallback)

    u_min = float(min(axis_u) - pad_u)
    u_max = float(max(axis_u) + pad_u)
    v_min = float(min(axis_v) - pad_v)
    v_max = float(max(axis_v) + pad_v)

    quad_proj = np.array(
        [[u_min, v_min], [u_max, v_min], [u_max, v_max], [u_min, v_max]],
        dtype=np.float32,
    )
    quad = quad_proj @ axes + center
    return order_quad_points(quad.reshape(-1, 2))


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
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def transform_circles(circles: Sequence[Tuple[float, float, float]], matrix: np.ndarray) -> List[Tuple[float, float, float]]:
    if not circles:
        return []

    points = np.array([[x, y] for x, y, _ in circles], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    median_radius = float(np.median([r for _, _, r in circles]))
    return [(float(x), float(y), median_radius) for x, y in transformed]


def detect_lego_size(image: np.ndarray) -> DetectionResult:
    height_result = detect_lego_height(image)
    mask = build_foreground_mask(image)
    bbox = largest_contour_bbox(mask)
    circles = detect_circles(image, bbox)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top_circles = select_top_face_circles(circles, gray, mask, bbox)

    if len(top_circles) < 4:
        return DetectionResult(
            dims=(0, 0),
            size_confidence=0.0,
            height=height_result.height,
            height_confidence=height_result.confidence,
            circles=circles,
            top_circles=[],
            bbox=bbox,
            top_face_quad=None,
            height_lines=height_result.layer_lines,
            message="No stable top studs detected. Try a clearer top view.",
        )

    top_face_quad = estimate_top_face_quad(top_circles)
    warped_circles = top_circles
    if top_face_quad is not None:
        _, matrix = warp_from_quad(image, top_face_quad)
        warped_circles = transform_circles(top_circles, matrix)

    dims, grid_confidence = infer_dims_from_centers(warped_circles)
    if dims == (0, 0):
        message = "Could not infer stud layout from top-face studs."
        size_confidence = 0.0
    else:
        message = f"Estimated stud layout: {dims[0]} x {dims[1]}"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = build_circle_features(top_circles, gray, mask, bbox)
        polarity_bonus = float(np.mean([max(0.0, feature.polarity) for feature in features])) if features else 0.0
        size_confidence = float(np.clip(0.85 * grid_confidence + 0.15 * min(1.0, polarity_bonus / 18.0), 0.0, 1.0))

    refined_height, refined_height_confidence = refine_compact_height(
        dims=dims,
        top_circles=top_circles,
        raw_height=height_result.height,
        raw_confidence=height_result.confidence,
    )

    return DetectionResult(
        dims=dims,
        size_confidence=size_confidence,
        height=refined_height,
        height_confidence=refined_height_confidence,
        circles=circles,
        top_circles=top_circles,
        bbox=bbox,
        top_face_quad=None if top_face_quad is None else [(float(x), float(y)) for x, y in top_face_quad],
        height_lines=height_result.layer_lines,
        message=message,
    )


def build_result_label(result: DetectionResult, height_only: bool = False, size_only: bool = False) -> str:
    parts: List[str] = []
    shape_3d = shape_3d_string(result)
    if shape_3d is not None and not height_only and not size_only:
        parts.append(f"shape={shape_3d}")
    if not height_only and result.dims != (0, 0):
        parts.append(f"size={result.dims[0]} x {result.dims[1]}")
        parts.append(f"size_conf={result.size_confidence:.2f}")
    if not size_only and result.height > 0:
        parts.append(f"height={result.height}")
        parts.append(f"height_conf={result.height_confidence:.2f}")
    if not parts:
        return result.message
    return "  ".join(parts)


def draw_result(
    image: np.ndarray,
    result: DetectionResult,
    height_only: bool = False,
    size_only: bool = False,
) -> np.ndarray:
    output = image.copy()
    bx, by, bw, bh = result.bbox
    cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (60, 180, 255), 2)

    if not size_only:
        for y in result.height_lines:
            cv2.line(output, (bx, y), (bx + bw, y), (255, 180, 0), 2)

    if not height_only:
        for x, y, r in result.circles:
            cv2.circle(output, (int(round(x)), int(round(y))), int(round(r)), (120, 120, 120), 1)

        if result.top_face_quad is not None:
            quad = np.array(result.top_face_quad, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [quad], True, (0, 220, 255), 2)

        for x, y, r in result.top_circles:
            cv2.circle(output, (int(round(x)), int(round(y))), int(round(r)), (0, 220, 0), 2)
            cv2.circle(output, (int(round(x)), int(round(y))), 2, (0, 220, 0), -1)

    label = build_result_label(result, height_only=height_only, size_only=size_only)

    cv2.putText(
        output,
        label,
        (20, max(30, by - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        (20, max(30, by - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate LEGO stud dimensions and visible height from an image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional annotated output image path. Defaults to <input>_detected.png.",
    )
    parser.add_argument(
        "--multi-stack",
        action="store_true",
        help="Detect multiple stack objects as 1 x 1 x layer-count.",
    )
    parser.add_argument(
        "--height-only",
        action="store_true",
        help="Only show visible height result and overlays.",
    )
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="Only show top-face size result and overlays.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.height_only and args.size_only:
        print("Choose only one of --height-only or --size-only.")
        return 1

    image_path = args.image
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    if args.multi_stack:
        if __package__:
            from .lego_multi_stack_detector import detect_multi_stack_objects, draw_result as draw_multi_stack_result
        else:
            from detectors.lego_multi_stack_detector import detect_multi_stack_objects, draw_result as draw_multi_stack_result

        objects = detect_multi_stack_objects(image)
        if not objects:
            print("No stack objects detected.")
            return 1

        for object_result in objects:
            print(f"Object {object_result.index}: 1 x 1 x {object_result.layers}")
            print(f"Confidence: {object_result.confidence:.2f}")

        default_output = image_path.with_name(f"{image_path.stem}_multi_stack.png")
        output_path = args.output or default_output
        annotated = draw_multi_stack_result(image, objects)
        cv2.imwrite(str(output_path), annotated)
        print(f"Annotated image saved to: {output_path}")
        return 0

    result = detect_lego_size(image)
    if not args.height_only and result.dims == (0, 0):
        print(result.message)
    elif not args.height_only:
        print(f"Detected size: {result.dims[0]} x {result.dims[1]}")
        print(f"Size confidence: {result.size_confidence:.2f}")

    if not args.size_only and result.height > 0:
        print(f"Detected visible height: {result.height} bricks")
        print(f"Height confidence: {result.height_confidence:.2f}")
    shape_3d = shape_3d_string(result)
    if shape_3d is not None:
        print(f"Estimated 3D shape: {shape_3d}")

    output_path = args.output or image_path.with_name(f"{image_path.stem}_detected.png")
    annotated = draw_result(image, result, height_only=args.height_only, size_only=args.size_only)
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
