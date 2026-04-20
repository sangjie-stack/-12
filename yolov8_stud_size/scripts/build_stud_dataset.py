from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lego_size_yolo.dataset_prep import build_crop_from_label, save_manifest


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build warped top-face crops for the stud detection dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("datasets/top_face_seg"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/stud_det"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--class-id", type=int, default=0, help="Top-face class id in the segmentation labels.")
    parser.add_argument("--quad-scale", type=float, default=1.04, help="Padding factor applied before perspective warp.")
    parser.add_argument("--preview-dir", type=Path, default=Path("runs/build_stud_dataset/previews"))
    return parser.parse_args()


def find_images(image_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(pattern)))
    return paths


def save_preview(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> int:
    args = parse_args()

    all_items = []
    for split in args.splits:
        image_dir = args.source_root / "images" / split
        label_dir = args.source_root / "labels" / split
        output_image_dir = args.output_root / "images" / split
        output_label_dir = args.output_root / "labels" / split

        images = find_images(image_dir)
        split_items = []
        for index, image_path in enumerate(images):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            output_image_path = output_image_dir / f"{image_path.stem}.png"
            output_label_path = output_label_dir / f"{image_path.stem}.txt"
            prepared = build_crop_from_label(
                image_path=image_path,
                label_path=label_path,
                output_image_path=output_image_path,
                output_label_path=output_label_path,
                split=split,
                index=index,
                target_class_id=args.class_id,
                quad_scale=args.quad_scale,
            )
            split_items.append(prepared)

            preview_path = args.preview_dir / split / f"{image_path.stem}.png"
            save_preview(output_image_path, preview_path)

        manifest_path = args.output_root / "manifests" / f"{split}.json"
        save_manifest(split_items, manifest_path)
        all_items.extend(split_items)
        print(f"[{split}] prepared {len(split_items)} crops")

    print(f"Total prepared crops: {len(all_items)}")
    print("Stud labels remain empty and need manual annotation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
