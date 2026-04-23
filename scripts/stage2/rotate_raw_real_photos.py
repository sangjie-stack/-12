import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_config import VALID_IMAGE_SUFFIXES


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate real weak-class photos directly under data/raw."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset root, for example data/raw.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["2x2", "2x3", "2x4"],
        help="Weak classes that should receive rotated real-photo copies.",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=[90.0, 180.0, 270.0],
        help="Rotation angles in degrees. Angle 0 is ignored automatically.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rotated files if they already exist.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="rotation_weak_real_summary.json",
        help="Summary JSON filename saved under raw-root.",
    )
    return parser.parse_args()


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


def is_real_photo_path(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    if "__rot_" in stem:
        return False
    if name.startswith("lemuwu_"):
        return False
    if name.startswith("photo_"):
        return True
    return stem.isdigit()


def iter_real_images(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        if is_real_photo_path(path):
            yield path


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    return image.convert("RGB").rotate(
        angle,
        resample=RESAMPLE_BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def build_rotated_path(image_path: Path, angle: float) -> Path:
    return image_path.with_name(f"{image_path.stem}__rot_{angle_token(angle)}.png")


def process_raw_dataset(
    raw_root: Path,
    class_names: Sequence[str],
    angles: Sequence[float],
    overwrite: bool,
    summary_name: str,
) -> Dict[str, object]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root does not exist: {raw_root}")
    if not angles:
        raise ValueError("No valid non-zero rotation angles were provided.")

    summary: Dict[str, object] = {
        "raw_root": str(raw_root.resolve()),
        "class_names": list(class_names),
        "angles": list(angles),
        "classes": {},
    }

    for class_name in class_names:
        class_dir = raw_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        original_real_count = 0
        created_count = 0
        skipped_existing = 0

        for image_path in iter_real_images(class_dir):
            original_real_count += 1
            with Image.open(image_path) as image:
                for angle in angles:
                    rotated_path = build_rotated_path(image_path, angle)
                    if rotated_path.exists() and not overwrite:
                        skipped_existing += 1
                        continue
                    rotated = rotate_image(image, angle)
                    rotated.save(rotated_path)
                    created_count += 1

        summary["classes"][class_name] = {
            "real_source_images": original_real_count,
            "created_rotated_images": created_count,
            "skipped_existing_images": skipped_existing,
        }

    summary_path = raw_root / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    angles = normalize_angles(args.angles)
    summary = process_raw_dataset(
        raw_root=raw_root,
        class_names=args.class_names,
        angles=angles,
        overwrite=args.overwrite,
        summary_name=args.summary_name,
    )
    print("Weak-class real-photo rotations created.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
