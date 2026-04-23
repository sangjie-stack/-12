from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from utils.dataset_config import BRICK_CLASSES, VALID_IMAGE_SUFFIXES


class LegoDataset(Dataset):
    def __init__(
        self,
        root: Path,
        class_names: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.class_names = list(class_names or BRICK_CLASSES)
        self.transform = transform
        self.class_to_index: Dict[str, int] = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        self.samples: List[Tuple[Path, int]] = self._scan_samples()

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split does not exist: {self.root}")

        samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                    samples.append((path, self.class_to_index[class_name]))
        if not samples:
            raise ValueError(f"No samples found under: {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label

