import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_config import BRICK_CLASSES, VALID_IMAGE_SUFFIXES


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a split dataset and create rotated image variants for training."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/splits_stage2_raw"),
        help="Source dataset root, for example data/splits_stage2_raw.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination dataset root for rotated outputs.",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=[90.0, 180.0, 270.0],
        help="Rotation angles in degrees. Angle 0 is ignored automatically.",
    )
    parser.add_argument(
        "--augment-splits",
        nargs="+",
        default=["train"],
        help="Split names that should receive rotated copies. Other splits are copied only.",
    )
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=None,
        help="Optional subset of split directories to process. Default: all directories under input-root.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Optional subset of class directories to copy into the output dataset.",
    )
    parser.add_argument(
        "--augment-class-names",
        nargs="+",
        default=None,
        help="Optional subset of class directories that should receive rotated copies. Other classes are copied only.",
    )
    parser.add_argument(
        "--rotate-source",
        choices=["all", "real"],
        default="all",
        help="Choose which images can be rotated. 'real' skips crawled lemuwu_* images and rotates real photos only.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional max number of original images to process per class. Useful for smoke tests.",
    )
    return parser.parse_args()


def ensure_valid_roots(input_root: Path, output_root: Path) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")
    if input_root.resolve() == output_root.resolve():
        raise ValueError("output-root must be different from input-root to avoid modifying the source dataset in place.")


def list_split_names(input_root: Path, requested: Optional[Sequence[str]] = None) -> List[str]:
    available = [path.name for path in sorted(input_root.iterdir()) if path.is_dir()]
    if requested is None:
        return available
    requested_set = list(dict.fromkeys(requested))
    missing = [name for name in requested_set if name not in available]
    if missing:
        raise ValueError(f"Requested splits do not exist under {input_root}: {', '.join(missing)}")
    return requested_set


def list_class_names(split_root: Path, requested: Optional[Sequence[str]] = None) -> List[str]:
    available = [path.name for path in sorted(split_root.iterdir()) if path.is_dir()]
    if requested is None:
        preferred = [class_name for class_name in BRICK_CLASSES if class_name in available]
        remaining = [class_name for class_name in available if class_name not in preferred]
        return preferred + remaining
    requested_set = list(dict.fromkeys(requested))
    missing = [name for name in requested_set if name not in available]
    if missing:
        raise ValueError(f"Requested classes do not exist under {split_root}: {', '.join(missing)}")
    return requested_set


def iter_images(class_dir: Path, limit: Optional[int] = None) -> Iterable[Path]:
    images = [
        path
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES and not path.name.startswith(".")
    ]
    if limit is not None:
        images = images[:limit]
    return images


def normalize_angles(angles: Sequence[float]) -> List[float]:
    normalized: List[float] = []
    seen: set[str] = set()
    for raw_angle in angles:
        angle = float(raw_angle) % 360.0
        if abs(angle) < 1e-8:
            continue
        key = f"{angle:.6f}"
        if key in seen:
            continue
        seen.add(key)
        normalized.append(angle)
    return normalized


def angle_token(angle: float) -> str:
    rounded = round(angle, 4)
    if float(rounded).is_integer():
        return f"{int(rounded):03d}"
    return str(rounded).replace(".", "_")


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    return image.convert("RGB").rotate(
        angle,
        resample=RESAMPLE_BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def is_real_photo_path(path: Path) -> bool:
    stem = path.stem.lower()
    name = path.name.lower()
    if name.startswith("lemuwu_"):
        return False
    if name.startswith("photo_"):
        return True
    return stem.isdigit()


def should_rotate_image(path: Path, rotate_source: str) -> bool:
    if rotate_source == "all":
        return True
    if rotate_source == "real":
        return is_real_photo_path(path)
    raise ValueError(f"Unsupported rotate_source: {rotate_source}")


def process_dataset(
    input_root: Path,
    output_root: Path,
    split_names: Sequence[str],
    augment_splits: Sequence[str],
    class_names: Optional[Sequence[str]],
    augment_class_names: Optional[Sequence[str]],
    rotate_source: str,
    angles: Sequence[float],
    limit_per_class: Optional[int],
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    augment_split_set = set(augment_splits)
    summary: Dict[str, object] = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "angles": list(angles),
        "augment_splits": list(augment_splits),
        "rotate_source": rotate_source,
        "splits": {},
    }

    for split_name in split_names:
        split_root = input_root / split_name
        split_output_root = output_root / split_name
        split_output_root.mkdir(parents=True, exist_ok=True)
        resolved_class_names = list_class_names(split_root, requested=class_names)
        augment_class_name_set = set(augment_class_names or resolved_class_names)

        split_original_count = 0
        split_rotated_count = 0
        split_class_summary: Dict[str, Dict[str, int]] = {}

        for class_name in resolved_class_names:
            source_class_dir = split_root / class_name
            target_class_dir = split_output_root / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            original_count = 0
            rotated_count = 0
            for image_path in iter_images(source_class_dir, limit=limit_per_class):
                target_original_path = target_class_dir / image_path.name
                shutil.copy2(image_path, target_original_path)
                original_count += 1

                if (
                    split_name not in augment_split_set
                    or class_name not in augment_class_name_set
                    or not should_rotate_image(image_path, rotate_source=rotate_source)
                ):
                    continue

                with Image.open(image_path) as image:
                    for angle in angles:
                        rotated = rotate_image(image, angle)
                        rotated_name = f"{image_path.stem}__rot_{angle_token(angle)}.png"
                        rotated.save(target_class_dir / rotated_name)
                        rotated_count += 1

            split_original_count += original_count
            split_rotated_count += rotated_count
            split_class_summary[class_name] = {
                "original_images": original_count,
                "rotated_images": rotated_count,
                "total_output_images": original_count + rotated_count,
            }

        summary["splits"][split_name] = {
            "original_images": split_original_count,
            "rotated_images": split_rotated_count,
            "total_output_images": split_original_count + split_rotated_count,
            "classes": split_class_summary,
        }

    summary_path = output_root / "rotation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    ensure_valid_roots(input_root, output_root)

    split_names = list_split_names(input_root, requested=args.include_splits)
    angles = normalize_angles(args.angles)
    if not angles and any(split in set(args.augment_splits) for split in split_names):
        raise ValueError("No valid rotation angles remain after normalization. Please provide a non-zero angle.")

    summary = process_dataset(
        input_root=input_root,
        output_root=output_root,
        split_names=split_names,
        augment_splits=args.augment_splits,
        class_names=args.class_names,
        augment_class_names=args.augment_class_names,
        rotate_source=args.rotate_source,
        angles=angles,
        limit_per_class=args.limit_per_class,
    )
    print("Rotated dataset created.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
