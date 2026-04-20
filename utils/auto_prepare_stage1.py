import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from dataset_config import BRICK_CLASSES


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
SPLITS_ROOT = ROOT / "data" / "splits"
QUALITY_REPORT = ROOT / "data" / "quality_report.json"
SPLIT_REPORT = ROOT / "data" / "splits" / "split_report.json"
SUMMARY_REPORT = ROOT / "assets" / "stage1_auto_summary.md"


def ensure_structure() -> None:
    directories = [
        ROOT / "model",
        ROOT / "utils",
        ROOT / "pages",
        ROOT / "data",
        RAW_ROOT,
        PROCESSED_ROOT,
        SPLITS_ROOT,
        ROOT / "assets",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    for class_name in BRICK_CLASSES:
        (RAW_ROOT / class_name).mkdir(parents=True, exist_ok=True)


def run_step(command: List[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def build_summary() -> str:
    quality = json.loads(QUALITY_REPORT.read_text(encoding="utf-8")) if QUALITY_REPORT.exists() else {}
    split = json.loads(SPLIT_REPORT.read_text(encoding="utf-8")) if SPLIT_REPORT.exists() else {}

    lines = [
        "# Stage 1 Auto Summary",
        "",
        "## Dataset Overview",
        "",
        f"- Dataset root: `{quality.get('dataset_root', 'data/raw')}`",
        f"- Class count: {quality.get('class_count', 0)}",
        f"- Total valid images: {quality.get('dimension_summary', {}).get('total_images', 0)}",
        f"- Invalid files: {quality.get('invalid_file_count', 0)}",
    ]

    missing_classes = quality.get("missing_classes", [])
    extra_classes = quality.get("extra_classes", [])
    if missing_classes:
        lines.append(f"- Missing classes: {', '.join(missing_classes)}")
    else:
        lines.append("- Missing classes: none")
    if extra_classes:
        lines.append(f"- Extra classes: {', '.join(extra_classes)}")
    else:
        lines.append("- Extra classes: none")

    lines.extend(["", "## Per-class Counts", ""])
    for class_name, count in quality.get("per_class_counts", {}).items():
        lines.append(f"- {class_name}: {count}")

    lines.extend(["", "## Split Summary", ""])
    for split_name in ("train", "val", "test"):
        split_counts = split.get(split_name, {})
        split_total = sum(split_counts.values())
        lines.append(f"- {split_name}: {split_total}")
        for class_name in BRICK_CLASSES:
            if class_name in split_counts:
                lines.append(f"  - {class_name}: {split_counts[class_name]}")

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- Quality report: `{QUALITY_REPORT.relative_to(ROOT)}`",
            f"- Split report: `{SPLIT_REPORT.relative_to(ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full stage-1 dataset preparation workflow.")
    parser.add_argument("--size", type=int, default=64, help="Target image size for preprocessing.")
    parser.add_argument("--prefix", type=str, default="lego", help="Filename prefix after renaming.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate synthetic Brick sample images for the six base classes before preprocessing.",
    )
    parser.add_argument("--samples-per-class", type=int, default=20, help="Generated sample count for each class.")
    parser.add_argument("--raw-image-size", type=int, default=320, help="Generated raw image size.")
    args = parser.parse_args()

    ensure_structure()
    if args.generate_samples:
        run_step(
            [
                sys.executable,
                "utils/generate_brick_samples.py",
                "data/raw",
                "--samples-per-class",
                str(args.samples_per_class),
                "--image-size",
                str(args.raw_image_size),
                "--seed",
                str(args.seed),
                "--clean",
            ]
        )
    run_step(
        [
            sys.executable,
            "utils/check_dataset_quality.py",
            "data/raw",
            "--output",
            str(QUALITY_REPORT.relative_to(ROOT)),
        ]
    )
    run_step(
        [
            sys.executable,
            "utils/resize_and_rename.py",
            "data/raw",
            "data/processed",
            "--size",
            str(args.size),
            "--prefix",
            args.prefix,
            "--clean",
        ]
    )
    run_step(
        [
            sys.executable,
            "utils/split_dataset.py",
            "data/processed",
            "data/splits",
            "--seed",
            str(args.seed),
            "--clean",
            "--report",
            str(SPLIT_REPORT.relative_to(ROOT)),
        ]
    )

    summary_text = build_summary()
    SUMMARY_REPORT.write_text(summary_text, encoding="utf-8")
    print(f"一键流程执行完成，阶段摘要已生成: {SUMMARY_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
