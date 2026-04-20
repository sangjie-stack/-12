from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector for LEGO top studs.")
    parser.add_argument("--data", type=Path, default=Path("configs/stud_det.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("runs/stud_det"))
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
