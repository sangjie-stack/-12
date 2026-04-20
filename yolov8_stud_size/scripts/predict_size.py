from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lego_size_yolo import predict_size_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict LEGO stud layout with a two-stage YOLOv8 pipeline.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--top-face-weights", type=Path, required=True, help="YOLOv8 segmentation weights for top-face masks.")
    parser.add_argument("--stud-weights", type=Path, required=True, help="YOLOv8 detection weights for top studs.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict_size"), help="Directory for visualizations and JSON output.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--top-face-conf", type=float, default=0.25)
    parser.add_argument("--stud-conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction = predict_size_from_path(
        image_path=args.image,
        top_face_weights=args.top_face_weights,
        stud_weights=args.stud_weights,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        top_face_conf=args.top_face_conf,
        stud_conf=args.stud_conf,
    )

    rows, cols = prediction.canonical_size
    print(f"Predicted size: {rows} x {cols}")
    print(f"Confidence: {prediction.confidence:.2f}")
    print(f"Stud count: {prediction.stud_count}")
    print(f"Saved outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
