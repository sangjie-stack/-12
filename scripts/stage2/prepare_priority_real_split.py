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

from utils.dataset_config import BRICK_CLASSES, VALID_IMAGE_SUFFIXES, list_class_directories


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a base split dataset and duplicate selected real photos inside the train split."
    )
    parser.add_argument(
        "--base-split-root",
        type=Path,
        default=Path("data/splits_stage2_raw"),
        help="Existing train/val/test split to use as the base dataset.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset root used only when source-mode is raw_unseen.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination split root for the strengthened dataset.",
    )
    parser.add_argument(
        "--priority-classes",
        nargs="+",
        default=["1x1", "2x2", "2x4"],
        help="Classes whose train-split real photos should be emphasized.",
    )
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split directories to copy from the base split.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["split_train", "raw_unseen"],
        default="split_train",
        help="Where to source the reinforcement real photos from.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional max number of source real photos to inject per class.",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=[],
        help="Optional rotation angles for extra train-only copies of the injected photos.",
    )
    return parser.parse_args()


def ensure_valid_roots(base_split_root: Path, raw_root: Path, output_root: Path) -> None:
    if not base_split_root.exists():
        raise FileNotFoundError(f"Base split root does not exist: {base_split_root}")
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root does not exist: {raw_root}")
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}. "
            "Please choose a new directory to avoid overwriting an existing dataset."
        )


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
    name = path.name.lower()
    stem = path.stem.lower()
    if "__rot_" in stem:
        return False
    if name.startswith("lemuwu_"):
        return False
    if name.startswith("photo_"):
        return True
    return stem.isdigit()


def iter_class_images(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        yield path


def iter_real_images(class_dir: Path, limit: Optional[int] = None) -> Iterable[Path]:
    images = [path for path in iter_class_images(class_dir) if is_real_photo_path(path)]
    if limit is not None:
        images = images[:limit]
    return images


def copy_base_split(
    base_split_root: Path,
    output_root: Path,
    include_splits: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    available_splits = {path.name for path in base_split_root.iterdir() if path.is_dir()}
    missing_splits = [split_name for split_name in include_splits if split_name not in available_splits]
    if missing_splits:
        raise ValueError(
            f"Requested splits do not exist under {base_split_root}: {', '.join(missing_splits)}"
        )

    for split_name in include_splits:
        split_root = base_split_root / split_name
        split_output_root = output_root / split_name
        summary[split_name] = {}
        for class_dir in list_class_directories(split_root):
            target_class_dir = split_output_root / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for image_path in iter_class_images(class_dir):
                shutil.copy2(image_path, target_class_dir / image_path.name)
                copied += 1
            summary[split_name][class_dir.name] = copied
    return summary


def inject_priority_real_images(
    base_split_root: Path,
    raw_root: Path,
    output_root: Path,
    priority_classes: Sequence[str],
    include_splits: Sequence[str],
    source_mode: str,
    angles: Sequence[float],
    limit_per_class: Optional[int],
) -> Dict[str, Dict[str, int]]:
    train_root = output_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    for class_name in priority_classes:
        source_class_dir = (base_split_root / "train" / class_name) if source_mode == "split_train" else (raw_root / class_name)
        if not source_class_dir.exists():
            raise FileNotFoundError(f"Priority class directory does not exist under source root: {source_class_dir}")

        target_class_dir = train_root / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        added_original = 0
        added_rotated = 0
        source_count = 0
        existing_split_names = set()
        if source_mode == "raw_unseen":
            for split_name in include_splits:
                existing_class_dir = base_split_root / split_name / class_name
                if not existing_class_dir.exists():
                    continue
                existing_split_names.update(path.name.lower() for path in iter_class_images(existing_class_dir))

        source_images = list(iter_real_images(source_class_dir, limit=limit_per_class))
        if source_mode == "raw_unseen":
            source_images = [path for path in source_images if path.name.lower() not in existing_split_names]

        for image_path in source_images:
            source_count += 1
            shutil.copy2(image_path, target_class_dir / f"priority_real__{image_path.name}")
            added_original += 1

            if not angles:
                continue

            with Image.open(image_path) as image:
                for angle in angles:
                    rotated = rotate_image(image, angle)
                    rotated_name = f"priority_real__{image_path.stem}__rot_{angle_token(angle)}.png"
                    rotated.save(target_class_dir / rotated_name)
                    added_rotated += 1

        summary[class_name] = {
            "source_mode": source_mode,
            "source_real_images": source_count,
            "added_original_images": added_original,
            "added_rotated_images": added_rotated,
            "total_added_images": added_original + added_rotated,
        }
    return summary


def build_split_report(output_root: Path) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {}
    for split_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        report[split_dir.name] = {}
        for class_name in BRICK_CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            report[split_dir.name][class_name] = sum(1 for _ in iter_class_images(class_dir))
    return report


def main() -> int:
    args = parse_args()
    base_split_root = args.base_split_root.resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    priority_classes = list(dict.fromkeys(args.priority_classes))
    angles = normalize_angles(args.angles)

    ensure_valid_roots(base_split_root, raw_root, output_root)
    base_copy_summary = copy_base_split(
        base_split_root=base_split_root,
        output_root=output_root,
        include_splits=args.include_splits,
    )
    injection_summary = inject_priority_real_images(
        raw_root=raw_root,
        base_split_root=base_split_root,
        output_root=output_root,
        priority_classes=priority_classes,
        include_splits=args.include_splits,
        source_mode=args.source_mode,
        angles=angles,
        limit_per_class=args.limit_per_class,
    )
    split_report = build_split_report(output_root)

    summary = {
        "base_split_root": str(base_split_root),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "priority_classes": priority_classes,
        "source_mode": args.source_mode,
        "angles": angles,
        "base_copy_summary": base_copy_summary,
        "injection_summary": injection_summary,
        "split_report": split_report,
    }
    (output_root / "priority_real_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "split_report.json").write_text(
        json.dumps(split_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Priority real-photo split created.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
