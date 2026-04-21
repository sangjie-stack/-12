from pathlib import Path
from typing import Iterable, List


BRICK_CLASSES = ["1x1", "1x2", "1x3", "1x4", "2x2", "2x3", "2x4"]
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_FILENAMES = {".gitkeep"}


def list_class_directories(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not should_ignore_file(path) and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
            yield path


def should_ignore_file(path: Path) -> bool:
    return path.name in IGNORED_FILENAMES or path.name.startswith(".")
