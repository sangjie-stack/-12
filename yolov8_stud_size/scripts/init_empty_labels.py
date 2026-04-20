from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create empty YOLO label files for images without labels.")
    parser.add_argument("--images", type=Path, required=True, help="Image directory.")
    parser.add_argument("--labels", type=Path, required=True, help="Label directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.labels.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for image_path in sorted(args.images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = args.labels / f"{image_path.stem}.txt"
        if label_path.exists():
            skipped += 1
            continue
        label_path.touch()
        created += 1

    print(f"Created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Labels dir: {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
