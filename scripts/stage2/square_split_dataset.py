import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_config import VALID_IMAGE_SUFFIXES
from utils.image_preprocess import auto_crop_to_square, reduce_white_background_shadows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an existing train/val/test split dataset into square images."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/splits_stage2_raw"),
        help="Source split dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination split dataset root for square images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Square output size, for example 128.",
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Crop the brick foreground to square before resizing.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing output root before writing new files.",
    )
    parser.add_argument(
        "--reduce-shadow",
        action="store_true",
        help="Lighten neutral white-background shadows before square resizing.",
    )
    parser.add_argument(
        "--shadow-brightness-floor",
        type=int,
        default=150,
        help="Minimum background brightness considered for shadow reduction.",
    )
    parser.add_argument(
        "--shadow-neutral-threshold",
        type=int,
        default=20,
        help="Maximum RGB channel spread treated as neutral background.",
    )
    parser.add_argument(
        "--shadow-protect-threshold",
        type=int,
        default=235,
        help="Pixels darker than this on any channel are protected as probable foreground.",
    )
    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=0.85,
        help="Blend strength used to whiten neutral shadows.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Convert images to black/white masks after square resizing.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="RGB white-background threshold used for binarization.",
    )
    return parser.parse_args()


def clear_output_root(root: Path) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def make_binary_image(image: Image.Image, white_threshold: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    source = rgb.load()
    binary = Image.new("RGB", (width, height), color=(255, 255, 255))
    target = binary.load()
    for y in range(height):
        for x in range(width):
            pixel = source[x, y]
            is_foreground = any(channel < white_threshold for channel in pixel)
            target[x, y] = (0, 0, 0) if is_foreground else (255, 255, 255)
    return binary


def make_square_image(
    image: Image.Image,
    size: int,
    auto_crop: bool,
    reduce_shadow: bool,
    shadow_brightness_floor: int,
    shadow_neutral_threshold: int,
    shadow_protect_threshold: int,
    shadow_strength: float,
    binarize: bool,
    white_threshold: int,
) -> Image.Image:
    rgb = image.convert("RGB")
    if reduce_shadow:
        rgb = reduce_white_background_shadows(
            rgb,
            brightness_floor=shadow_brightness_floor,
            neutral_threshold=shadow_neutral_threshold,
            protect_white_threshold=shadow_protect_threshold,
            strength=shadow_strength,
        )
    if auto_crop:
        rgb = auto_crop_to_square(rgb)
    fitted = ImageOps.contain(rgb, (size, size))
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    offset_x = (size - fitted.width) // 2
    offset_y = (size - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    if binarize:
        return make_binary_image(canvas, white_threshold=white_threshold)
    return canvas


def iter_split_image_files(split_root: Path) -> List[Path]:
    return [
        path
        for path in sorted(split_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES and not path.name.startswith(".")
    ]


def process_split_dataset(
    input_root: Path,
    output_root: Path,
    size: int,
    auto_crop: bool,
    reduce_shadow: bool,
    shadow_brightness_floor: int,
    shadow_neutral_threshold: int,
    shadow_protect_threshold: int,
    shadow_strength: float,
    binarize: bool,
    white_threshold: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "size": size,
        "auto_crop": auto_crop,
        "reduce_shadow": reduce_shadow,
        "shadow_brightness_floor": shadow_brightness_floor,
        "shadow_neutral_threshold": shadow_neutral_threshold,
        "shadow_protect_threshold": shadow_protect_threshold,
        "shadow_strength": shadow_strength,
        "binarize": binarize,
        "white_threshold": white_threshold,
        "splits": {},
    }

    for split_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        split_name = split_dir.name
        split_processed = 0
        split_skipped = 0
        class_summary: Dict[str, Dict[str, int]] = {}

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            target_class_dir = output_root / split_name / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            processed = 0
            skipped = 0

            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES or image_path.name.startswith("."):
                    continue
                try:
                    with Image.open(image_path) as image:
                        squared = make_square_image(
                            image,
                            size=size,
                            auto_crop=auto_crop,
                            reduce_shadow=reduce_shadow,
                            shadow_brightness_floor=shadow_brightness_floor,
                            shadow_neutral_threshold=shadow_neutral_threshold,
                            shadow_protect_threshold=shadow_protect_threshold,
                            shadow_strength=shadow_strength,
                            binarize=binarize,
                            white_threshold=white_threshold,
                        )
                except (UnidentifiedImageError, OSError, ValueError):
                    skipped += 1
                    split_skipped += 1
                    continue

                output_path = target_class_dir / f"{image_path.stem}.png"
                squared.save(output_path, format="PNG")
                processed += 1
                split_processed += 1

            class_summary[class_dir.name] = {
                "processed_images": processed,
                "skipped_images": skipped,
            }

        summary["splits"][split_name] = {
            "processed_images": split_processed,
            "skipped_images": split_skipped,
            "classes": class_summary,
        }

    summary_path = output_root / "square_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input split dataset root does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("output-root must be different from input-root.")

    output_root.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clear_output_root(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    summary = process_split_dataset(
        input_root=input_root,
        output_root=output_root,
        size=args.size,
        auto_crop=args.auto_crop,
        reduce_shadow=args.reduce_shadow,
        shadow_brightness_floor=args.shadow_brightness_floor,
        shadow_neutral_threshold=args.shadow_neutral_threshold,
        shadow_protect_threshold=args.shadow_protect_threshold,
        shadow_strength=args.shadow_strength,
        binarize=args.binarize,
        white_threshold=args.white_threshold,
    )
    print("Square split dataset created.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
