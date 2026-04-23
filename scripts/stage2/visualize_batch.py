import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.training_utils import build_dataloaders, compute_mean_std, infer_class_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize one batch from the LEGO training dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data/splits"), help="Dataset split root.")
    parser.add_argument("--output", type=Path, default=Path("runs/stage2/batch_preview.png"), help="Output image path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--auto-crop", action="store_true", help="Auto-crop the brick foreground before resizing.")
    args = parser.parse_args()

    class_names = infer_class_names(args.data_root)
    mean, std = compute_mean_std(
        args.data_root / "train",
        image_size=args.image_size,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_crop=args.auto_crop,
    )
    loaders = build_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        augment=False,
        augmentation_policy="none",
        auto_crop=args.auto_crop,
        class_names=class_names,
    )
    images, labels = next(iter(loaders["train"]))

    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    images = images * std_tensor + mean_tensor
    images = torch.clamp(images, 0.0, 1.0)

    grid = make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Train batch labels: {labels.tolist()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Batch preview saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
