import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lego_cnn import create_lego_cnn_from_config
from model.training_utils import infer_checkpoint_class_names
from utils.image_preprocess import auto_crop_to_square


def build_transform(config: dict, mean, std):
    transform_list = []
    if config.get("auto_crop", False):
        transform_list.append(auto_crop_to_square)
    transform_list.extend(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transform_list)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a single LEGO image with a trained CNN checkpoint.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument("--class-names", nargs="+", default=None, help="Optional explicit class order for older checkpoints.")
    parser.add_argument("--top-k", type=int, default=6, help="How many class probabilities to print.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    class_names = infer_checkpoint_class_names(checkpoint, default_class_names=args.class_names)
    model = create_lego_cnn_from_config(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform(config, checkpoint["mean"], checkpoint["std"])
    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    pairs = sorted(zip(class_names, probs.tolist()), key=lambda item: item[1], reverse=True)
    print(f"pred_class= {pairs[0][0]}")
    for name, prob in pairs[: max(1, args.top_k)]:
        print(f"{name}: {prob:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
