import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataset_config import VALID_IMAGE_SUFFIXES, list_class_directories, should_ignore_file


def allocate_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    ratios = [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]
    raw = {name: total * ratio for name, ratio in ratios}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    for name, _ in sorted(ratios, key=lambda item: raw[item[0]] - int(raw[item[0]]), reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    if total >= 3:
        for name in ("train", "val", "test"):
            if counts[name] == 0:
                donor = max(counts, key=counts.get)
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] += 1
    return counts["train"], counts["val"], counts["test"]


def copy_files(paths: List[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)


def split_train_only_files(files: List[Path], train_only_marker: Optional[str]) -> Tuple[List[Path], List[Path]]:
    if not train_only_marker:
        return files, []
    regular_files = [path for path in files if train_only_marker not in path.stem]
    train_only_files = [path for path in files if train_only_marker in path.stem]
    return regular_files, train_only_files


def clear_split_root(root: Path) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a LEGO dataset into train/val/test folders.")
    parser.add_argument("input_root", type=Path, help="Input dataset root.")
    parser.add_argument("output_root", type=Path, help="Output split root.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing split folders before writing a new split result.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/splits/split_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--train-only-marker",
        type=str,
        default=None,
        help="Optional filename marker. Matching files are forced into the train split only.",
    )
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratio sum must equal 1.0")

    random.seed(args.seed)
    report: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    class_dirs = list_class_directories(args.input_root)
    if args.clean:
        clear_split_root(args.output_root)

    for class_dir in class_dirs:
        files = [
            path for path in sorted(class_dir.iterdir())
            if path.is_file() and not should_ignore_file(path) and path.suffix.lower() in VALID_IMAGE_SUFFIXES
        ]
        regular_files, train_only_files = split_train_only_files(files, args.train_only_marker)
        random.shuffle(regular_files)
        train_count, val_count, test_count = allocate_counts(
            len(regular_files), args.train_ratio, args.val_ratio, args.test_ratio
        )

        train_files = regular_files[:train_count] + train_only_files
        val_files = regular_files[train_count:train_count + val_count]
        test_files = regular_files[train_count + val_count:train_count + val_count + test_count]

        copy_files(train_files, args.output_root / "train" / class_dir.name)
        copy_files(val_files, args.output_root / "val" / class_dir.name)
        copy_files(test_files, args.output_root / "test" / class_dir.name)

        report["train"][class_dir.name] = len(train_files)
        report["val"][class_dir.name] = len(val_files)
        report["test"][class_dir.name] = len(test_files)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("划分完成。")
    for split_name in ("train", "val", "test"):
        split_total = sum(report[split_name].values())
        print(f"{split_name}: {split_total}")
    print(f"报告已保存到: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
