import argparse
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from dataset_config import VALID_IMAGE_SUFFIXES, list_class_directories, should_ignore_file


def normalize_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        fitted = ImageOps.contain(rgb_image, (size, size))
        canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
        offset_x = (size - fitted.width) // 2
        offset_y = (size - fitted.height) // 2
        canvas.paste(fitted, (offset_x, offset_y))
        return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize LEGO images to 64x64 and rename them consistently.")
    parser.add_argument("input_root", type=Path, help="Input dataset root.")
    parser.add_argument("output_root", type=Path, help="Output dataset root.")
    parser.add_argument("--size", type=int, default=64, help="Target image size.")
    parser.add_argument("--prefix", type=str, default="lego", help="Filename prefix.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing PNG files in each output class folder before writing new results.",
    )
    args = parser.parse_args()

    class_dirs = list_class_directories(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    for class_dir in class_dirs:
        target_dir = args.output_root / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for existing in target_dir.glob("*.png"):
                existing.unlink()
        index = 1
        for path in sorted(class_dir.iterdir()):
            if should_ignore_file(path):
                continue
            if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                skipped_count += 1
                continue
            try:
                resized = normalize_image(path, args.size)
            except (UnidentifiedImageError, OSError, ValueError):
                skipped_count += 1
                continue
            output_name = f"{args.prefix}_{class_dir.name}_{index:04d}.png"
            resized.save(target_dir / output_name, format="PNG")
            processed_count += 1
            index += 1

    print(f"处理完成，有效图片: {processed_count}，跳过文件: {skipped_count}")
    print(f"输出目录: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
