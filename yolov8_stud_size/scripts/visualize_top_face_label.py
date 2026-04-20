from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay a YOLO segmentation label on an image for QA.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--label", type=Path, required=True, help="YOLO segmentation label path.")
    parser.add_argument("--output", type=Path, required=True, help="Output overlay image path.")
    parser.add_argument("--line-color", type=int, nargs=3, default=(0, 220, 255), help="Polygon BGR color.")
    return parser.parse_args()


def read_label(label_path: Path) -> np.ndarray:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Label file is empty: {label_path}")

    first = lines[0].split()
    if len(first) < 7:
        raise ValueError("Segmentation label needs class id plus at least 3 points.")

    coords = [float(value) for value in first[1:]]
    if len(coords) % 2 != 0:
        raise ValueError("Coordinate count must be even.")
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2)


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    polygon = read_label(args.label)
    height, width = image.shape[:2]
    polygon[:, 0] *= width
    polygon[:, 1] *= height
    polygon_int = polygon.astype(np.int32).reshape(-1, 1, 2)

    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon_int], (30, 180, 255))
    blended = cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0)
    cv2.polylines(blended, [polygon_int], True, tuple(args.line_color), 3)

    for index, (x, y) in enumerate(polygon.astype(np.int32)):
        cv2.circle(blended, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(
            blended,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), blended)
    print(f"Saved overlay to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
