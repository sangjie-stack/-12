import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.training_utils import build_dataloaders, evaluate_loader, infer_checkpoint_class_names, infer_class_names
from model.lego_cnn import create_lego_cnn_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved LEGO CNN model on the test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/splits_stage2_raw"), help="Dataset split root.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="cpu", help="Device.")
    parser.add_argument("--class-names", nargs="+", default=None, help="Optional explicit class order for older checkpoints.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save evaluation metrics as JSON.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    candidate_class_names = args.class_names or infer_class_names(args.data_root)
    class_names = infer_checkpoint_class_names(checkpoint, default_class_names=candidate_class_names)

    loaders = build_dataloaders(
        data_root=args.data_root,
        image_size=config["image_size"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        augment=False,
        augmentation_policy="none",
        auto_crop=config.get("auto_crop", False),
        class_names=class_names,
    )
    model = create_lego_cnn_from_config(config, num_classes=len(class_names)).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_loader(model, loaders["test"], criterion, torch.device(args.device), class_names=class_names)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
