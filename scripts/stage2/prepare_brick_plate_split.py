import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_config import BRICK_CLASSES, PLATE_CLASSES, VALID_IMAGE_SUFFIXES


def copy_class_images(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    if not source_dir.exists():
        (target_dir / '.gitkeep').write_text('', encoding='utf-8')
        return copied
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
            shutil.copy2(path, target_dir / path.name)
            copied += 1
    if copied == 0:
        (target_dir / '.gitkeep').write_text('', encoding='utf-8')
    return copied


def build_split(brick_root: Path, target_root: Path, plate_root: Optional[Path] = None) -> dict:
    split_report = {}
    for split_name in ('train', 'val', 'test'):
        split_report[split_name] = {}
        for class_name in BRICK_CLASSES:
            source_dir = brick_root / split_name / class_name
            target_dir = target_root / split_name / class_name
            split_report[split_name][class_name] = copy_class_images(source_dir, target_dir)
        for class_name in PLATE_CLASSES:
            source_dir = plate_root / split_name / class_name if plate_root is not None else Path('__missing__')
            target_dir = target_root / split_name / class_name
            split_report[split_name][class_name] = copy_class_images(source_dir, target_dir)
    return split_report


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a Brick+Plate split scaffold by copying the current Brick split and adding Plate folders.')
    parser.add_argument('--brick-root', type=Path, default=Path('data/splits_stage2_raw'))
    parser.add_argument('--target-root', type=Path, default=Path('data/splits_brick_plate'))
    parser.add_argument('--plate-root', type=Path, default=None, help='Optional split root containing prepared Plate train/val/test folders.')
    args = parser.parse_args()

    brick_root = (ROOT / args.brick_root).resolve() if not args.brick_root.is_absolute() else args.brick_root
    target_root = (ROOT / args.target_root).resolve() if not args.target_root.is_absolute() else args.target_root
    plate_root = None
    if args.plate_root is not None:
        plate_root = (ROOT / args.plate_root).resolve() if not args.plate_root.is_absolute() else args.plate_root

    target_root.mkdir(parents=True, exist_ok=True)
    split_report = build_split(brick_root, target_root, plate_root=plate_root)
    (target_root / 'split_report.json').write_text(json.dumps(split_report, ensure_ascii=False, indent=2), encoding='utf-8')
    (target_root / 'README.md').write_text(
        'Brick+Plate joint split scaffold\n\n'
        'This directory combines the current 7 Brick classes with 3 Plate classes: '
        'plate_2x2, plate_2x4, plate_4x4.\n'
        'Before training the joint model, add real Plate images into the matching '
        'train/val/test class folders.\n',
        encoding='utf-8',
    )
    print(target_root)
    print(json.dumps(split_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
