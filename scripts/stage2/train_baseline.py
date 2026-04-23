import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lego_cnn import DEFAULT_CONV_POOL_DEPTH, DEFAULT_MODEL_VARIANT, MODEL_VARIANTS
from model.training_utils import AUGMENTATION_POLICIES, TrainConfig, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the LegoCNN classifier.")
    parser.add_argument("--data-root", type=str, default="data/splits", help="Dataset split root.")
    parser.add_argument("--output-dir", type=str, default="runs/stage2/baseline", help="Training output directory.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout ratio.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count.")
    parser.add_argument("--depth", type=int, default=DEFAULT_CONV_POOL_DEPTH, help="Number of convolution stages.")
    parser.add_argument(
        "--model-variant",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        choices=MODEL_VARIANTS,
        help="Model architecture variant.",
    )
    parser.add_argument("--blocks-per-stage", type=int, default=2, help="Residual blocks per stage for deep models.")
    parser.add_argument("--fc-hidden-dim", type=int, default=256, help="Hidden dimension of the classifier head.")
    parser.add_argument("--disable-fc-batch-norm", action="store_true", help="Disable BatchNorm1d in the classifier head.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--augment", action="store_true", help="Legacy shorthand for --augmentation-policy affine.")
    parser.add_argument(
        "--augmentation-policy",
        type=str,
        default="none",
        choices=AUGMENTATION_POLICIES,
        help="Train-split augmentation policy.",
    )
    parser.add_argument("--auto-crop", action="store_true", help="Auto-crop the brick foreground before resizing.")
    parser.add_argument(
        "--shadow-reduce-probability",
        type=float,
        default=0.0,
        help="Probability of applying white-background shadow reduction to a training image.",
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
        "--priority-classes",
        nargs="+",
        default=[],
        help="Class names to emphasize during training, for example 1x2 1x4.",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=1.0,
        help="Extra loss weight for priority classes. Use values above 1.0 to emphasize them.",
    )
    parser.add_argument(
        "--priority-oversample",
        action="store_true",
        help="Oversample priority classes in the training loader.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to warm-start from before training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device, for example cpu or cuda.")
    args = parser.parse_args()

    augmentation_policy = args.augmentation_policy
    if args.augment and augmentation_policy == "none":
        augmentation_policy = "affine"

    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        optimizer=args.optimizer,
        base_channels=args.base_channels,
        depth=args.depth,
        model_variant=args.model_variant,
        blocks_per_stage=args.blocks_per_stage,
        fc_hidden_dim=args.fc_hidden_dim,
        fc_batch_norm=not args.disable_fc_batch_norm,
        patience=args.patience,
        num_workers=args.num_workers,
        augment=augmentation_policy != "none",
        augmentation_policy=augmentation_policy,
        auto_crop=args.auto_crop,
        shadow_reduce_probability=args.shadow_reduce_probability,
        shadow_brightness_floor=args.shadow_brightness_floor,
        shadow_neutral_threshold=args.shadow_neutral_threshold,
        shadow_protect_threshold=args.shadow_protect_threshold,
        shadow_strength=args.shadow_strength,
        priority_classes=tuple(args.priority_classes),
        priority_weight=args.priority_weight,
        priority_oversample=args.priority_oversample,
        init_checkpoint=args.init_checkpoint,
        seed=args.seed,
        device=args.device,
    )
    result = train_model(config, output_dir=Path(args.output_dir))
    print("Training finished.")
    print(result["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


