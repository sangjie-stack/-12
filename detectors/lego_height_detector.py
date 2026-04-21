import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class HeightResult:
    height: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    layer_lines: List[int]
    message: str


def build_foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colorful = hsv[:, :, 1] > 30
    darker = gray < 245
    mask = np.where(colorful | darker, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def largest_contour_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, mask.shape[1], mask.shape[0]
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def smooth_profile(values: np.ndarray, window: int = 9) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def cluster_peaks(peaks: Sequence[Tuple[int, float, int]], tolerance: int = 8) -> List[Tuple[int, float, int]]:
    if not peaks:
        return []

    ordered = sorted(peaks, key=lambda item: item[0])
    groups: List[List[Tuple[int, float, int]]] = [[ordered[0]]]
    for peak in ordered[1:]:
        if abs(peak[0] - groups[-1][-1][0]) <= tolerance:
            groups[-1].append(peak)
        else:
            groups.append([peak])

    clustered = []
    for group in groups:
        scores = np.array([score for _, score, _ in group], dtype=np.float32)
        ys = np.array([y for y, _, _ in group], dtype=np.float32)
        widths = np.array([width for _, _, width in group], dtype=np.float32)
        y = int(round(float(np.average(ys, weights=np.maximum(scores, 1e-3)))))
        score = float(np.max(scores))
        width = int(round(float(np.max(widths))))
        clustered.append((y, score, width))
    return clustered


def find_horizontal_layer_peaks(gray: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, float, int]]:
    bx, by, bw, bh = bbox
    vertical_grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    mask_bool = mask > 0
    width_profile = mask_bool.sum(axis=1).astype(np.float32)
    profile = (vertical_grad * mask_bool).sum(axis=1) / (width_profile + 1.0)
    profile = smooth_profile(profile, window=9)

    threshold = max(8.0, float(np.percentile(profile[by: by + bh], 72)))
    peaks: List[Tuple[int, float, int]] = []
    for y in range(max(2, by + 2), min(len(profile) - 2, by + bh - 2)):
        if width_profile[y] < max(20.0, bw * 0.10):
            continue
        if profile[y] < threshold:
            continue
        if profile[y] >= profile[y - 1] and profile[y] >= profile[y + 1]:
            peaks.append((y, float(profile[y]), int(width_profile[y])))
    return cluster_peaks(peaks, tolerance=8)


def choose_staircase_layers(peaks: Sequence[Tuple[int, float, int]], bbox: Tuple[int, int, int, int]) -> List[Tuple[int, float, int]]:
    bx, by, bw, bh = bbox
    filtered = [peak for peak in peaks if peak[2] >= max(24, int(bw * 0.10))]
    filtered = [peak for peak in filtered if peak[0] <= by + bh - 14]
    if len(filtered) < 2:
        return filtered

    filtered.sort(key=lambda item: item[0])

    monotonic: List[Tuple[int, float, int]] = [filtered[0]]
    for peak in filtered[1:]:
        last_y, _, last_width = monotonic[-1]
        y, score, width = peak
        if y - last_y < 20:
            if score > monotonic[-1][1]:
                monotonic[-1] = peak
            continue
        if width + 8 < last_width:
            continue
        monotonic.append(peak)

    if len(monotonic) >= 3:
        diffs = np.diff(np.array([item[0] for item in monotonic], dtype=np.float32))
        median_step = float(np.median(diffs))
        if len(monotonic) >= 2 and monotonic[0][2] < 0.75 * monotonic[1][2]:
            first_gap = monotonic[1][0] - monotonic[0][0]
            if abs(first_gap - median_step) <= max(10.0, median_step * 0.35):
                monotonic = monotonic[1:]

    return monotonic


def estimate_height_from_layers(layers: Sequence[Tuple[int, float, int]]) -> Tuple[int, float]:
    if not layers:
        return 0, 0.0
    if len(layers) == 1:
        return 1, 0.35

    ys = np.array([item[0] for item in layers], dtype=np.float32)
    widths = np.array([item[2] for item in layers], dtype=np.float32)
    diffs = np.diff(ys)
    if len(diffs) == 0:
        return 1, 0.35

    mean_step = float(np.mean(diffs))
    regularity = 1.0 if len(diffs) == 1 else max(0.0, 1.0 - float(np.std(diffs) / max(mean_step, 1e-6)))
    width_trend = 1.0 if len(widths) == 1 else max(0.0, 1.0 - float(np.mean(np.maximum(0.0, widths[:-1] - widths[1:])) / max(np.mean(widths), 1e-6)))
    confidence = float(np.clip(0.65 * regularity + 0.35 * width_trend, 0.0, 1.0))
    return len(layers), confidence


def detect_lego_height(image: np.ndarray) -> HeightResult:
    mask = build_foreground_mask(image)
    bbox = largest_contour_bbox(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    peaks = find_horizontal_layer_peaks(gray, mask, bbox)
    layers = choose_staircase_layers(peaks, bbox)
    height, confidence = estimate_height_from_layers(layers)

    if height == 0:
        message = "Could not estimate brick height."
    else:
        message = f"Estimated visible height: {height} bricks"

    return HeightResult(
        height=height,
        confidence=confidence,
        bbox=bbox,
        layer_lines=[y for y, _, _ in layers],
        message=message,
    )


def draw_result(image: np.ndarray, result: HeightResult) -> np.ndarray:
    output = image.copy()
    bx, by, bw, bh = result.bbox
    cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (60, 180, 255), 2)

    for y in result.layer_lines:
        cv2.line(output, (bx, y), (bx + bw, y), (0, 220, 0), 2)

    label = result.message
    if result.height > 0:
        label += f"  confidence={result.confidence:.2f}"

    cv2.putText(
        output,
        label,
        (20, max(30, by - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        (20, max(30, by - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate visible LEGO brick height from an image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional annotated output image path. Defaults to <input>_height.png.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return 1

    result = detect_lego_height(image)
    if result.height == 0:
        print(result.message)
    else:
        print(f"Detected visible height: {result.height} bricks")
        print(f"Confidence: {result.confidence:.2f}")

    output_path = args.output or image_path.with_name(f"{image_path.stem}_height.png")
    annotated = draw_result(image, result)
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
