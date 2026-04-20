import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image, UnidentifiedImageError

from dataset_config import BRICK_CLASSES, VALID_IMAGE_SUFFIXES, list_class_directories, should_ignore_file


def inspect_image(path: Path) -> Dict[str, object]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        return {
            "path": str(path),
            "format": image.format,
            "mode": image.mode,
            "width": width,
            "height": height,
        }


def build_report(dataset_root: Path) -> Dict[str, object]:
    class_dirs = list_class_directories(dataset_root)
    invalid_files: List[Dict[str, str]] = []
    class_counts: Dict[str, int] = {}
    format_counts: Counter = Counter()
    widths: List[int] = []
    heights: List[int] = []
    modes: Counter = Counter()
    per_class_sizes = defaultdict(list)

    for class_dir in class_dirs:
        count = 0
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if should_ignore_file(path):
                continue
            if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                invalid_files.append({"path": str(path), "reason": "unsupported extension"})
                continue
            try:
                info = inspect_image(path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid_files.append({"path": str(path), "reason": str(exc)})
                continue

            count += 1
            format_counts.update([str(info["format"])])
            widths.append(int(info["width"]))
            heights.append(int(info["height"]))
            modes.update([str(info["mode"])])
            per_class_sizes[class_dir.name].append([int(info["width"]), int(info["height"])])
        class_counts[class_dir.name] = count

    present_classes = {path.name for path in class_dirs}
    missing_classes = [name for name in BRICK_CLASSES if name not in present_classes]
    extra_classes = [name for name in sorted(present_classes) if name not in BRICK_CLASSES]

    report: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "class_count": len(class_dirs),
        "classes": [path.name for path in class_dirs],
        "expected_classes": BRICK_CLASSES,
        "missing_classes": missing_classes,
        "extra_classes": extra_classes,
        "per_class_counts": class_counts,
        "invalid_files": invalid_files,
        "invalid_file_count": len(invalid_files),
        "format_distribution": dict(format_counts),
        "mode_distribution": dict(modes),
        "dimension_summary": {
            "total_images": sum(class_counts.values()),
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
        },
        "per_class_example_sizes": {
            class_name: sizes[:5] for class_name, sizes in sorted(per_class_sizes.items())
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LEGO dataset quality and export a JSON report.")
    parser.add_argument("dataset_root", type=Path, help="Input dataset root. Expected layout: root/class_name/image.png")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/quality_report.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    report = build_report(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"检查完成，数据集根目录: {report['dataset_root']}")
    print(f"类别数量: {report['class_count']}")
    print(f"有效图片总数: {report['dimension_summary']['total_images']}")
    print(f"无效文件数量: {report['invalid_file_count']}")
    if report["missing_classes"]:
        print(f"缺失类别: {', '.join(report['missing_classes'])}")
    if report["extra_classes"]:
        print(f"额外类别: {', '.join(report['extra_classes'])}")
    for class_name, count in report["per_class_counts"].items():
        print(f"{class_name}: {count}")
    print(f"报告已保存到: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
