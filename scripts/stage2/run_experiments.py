import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lego_cnn import DEFAULT_CONV_POOL_DEPTH, DEFAULT_MODEL_VARIANT, MODEL_VARIANTS
from model.training_utils import AUGMENTATION_POLICIES, TrainConfig, train_model


EXPERIMENT_PRESETS: Dict[str, List[Dict[str, object]]] = {
    "batch_size": [
        {"batch_size": 8},
        {"batch_size": 16},
        {"batch_size": 32},
        {"batch_size": 64},
    ],
    "learning_rate": [
        {"learning_rate": 1e-2},
        {"learning_rate": 1e-3},
        {"learning_rate": 1e-4},
    ],
    "epoch": [
        {"epochs": 5},
        {"epochs": 10},
        {"epochs": 20},
    ],
    "dropout": [
        {"dropout": 0.0},
        {"dropout": 0.3},
        {"dropout": 0.5},
    ],
    "optimizer": [
        {"optimizer": "sgd"},
        {"optimizer": "adam"},
        {"optimizer": "adamw"},
    ],
    "depth_width": [
        {"depth": 2, "base_channels": 16},
        {"depth": 3, "base_channels": 32},
        {"depth": 4, "base_channels": 48},
    ],
    "augmentation": [
        {"augmentation_policy": "none"},
        {"augmentation_policy": "affine"},
        {"augmentation_policy": "affine_flip"},
        {"augmentation_policy": "affine_color"},
    ],
}


def format_override_label(override: Dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in override.items())


def write_summary_markdown(output_root: Path, experiment: str, summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        return

    best_val_row = min(summary_rows, key=lambda row: row["metrics"]["best_val_loss"])
    best_acc_row = max(summary_rows, key=lambda row: row["metrics"]["test_accuracy"])
    lines = [
        f"# {experiment} experiment summary",
        "",
        f"- Best validation loss: `{best_val_row['run_name']}` -> `{best_val_row['metrics']['best_val_loss']:.4f}`",
        f"- Best test accuracy: `{best_acc_row['run_name']}` -> `{best_acc_row['metrics']['test_accuracy']:.4f}`",
        "",
        "| Run | Override | Best epoch | Best val loss | Test accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        metrics = row["metrics"]
        lines.append(
            "| {run_name} | {override} | {best_epoch} | {best_val_loss:.4f} | {test_accuracy:.4f} |".format(
                run_name=row["run_name"],
                override=format_override_label(row["override"]),
                best_epoch=metrics["best_epoch"],
                best_val_loss=metrics["best_val_loss"],
                test_accuracy=metrics["test_accuracy"],
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_summary(output_root: Path, experiment: str, summary_rows: List[Dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_markdown(output_root, experiment, summary_rows)

    if not summary_rows:
        return

    x_labels = []
    val_losses = []
    test_accuracies = []
    for row in summary_rows:
        label = format_override_label(row["override"])
        x_labels.append(label)
        val_losses.append(row["metrics"]["best_val_loss"])
        test_accuracies.append(row["metrics"]["test_accuracy"])

    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, val_losses, marker="o")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Best Val Loss")
    plt.title(f"{experiment} comparison: validation loss")
    plt.tight_layout()
    plt.savefig(output_root / "summary_val_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, test_accuracies, marker="o")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title(f"{experiment} comparison: test accuracy")
    plt.tight_layout()
    plt.savefig(output_root / "summary_test_accuracy.png", dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LEGO training hyperparameter experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="batch_size",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        help="Experiment family to run.",
    )
    parser.add_argument("--data-root", type=str, default="data/splits", help="Dataset split root.")
    parser.add_argument("--output-root", type=str, default="runs/stage2/experiments", help="Experiment output root.")
    parser.add_argument("--epochs", type=int, default=15, help="Base epoch count for each run.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Base batch size for non-varied settings.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Base learning rate for non-varied settings.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Base weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Base dropout ratio for non-varied settings.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="Base optimizer for non-varied settings.",
    )
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count for non-varied settings.")
    parser.add_argument("--depth", type=int, default=DEFAULT_CONV_POOL_DEPTH, help="Base model depth for non-varied settings.")
    parser.add_argument(
        "--model-variant",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        choices=MODEL_VARIANTS,
        help="Base architecture for non-varied settings.",
    )
    parser.add_argument("--blocks-per-stage", type=int, default=2, help="Residual blocks per stage for non-varied settings.")
    parser.add_argument("--fc-hidden-dim", type=int, default=256, help="Classifier hidden dimension for non-varied settings.")
    parser.add_argument("--disable-fc-batch-norm", action="store_true", help="Disable BatchNorm1d in the classifier head.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--augmentation-policy",
        type=str,
        default="none",
        choices=AUGMENTATION_POLICIES,
        help="Base augmentation policy for non-varied settings.",
    )
    parser.add_argument("--auto-crop", action="store_true", help="Auto-crop the brick foreground before resizing.")
    parser.add_argument(
        "--priority-classes",
        nargs="+",
        default=[],
        help="Optional priority classes to emphasize during every run.",
    )
    parser.add_argument("--priority-weight", type=float, default=1.0, help="Loss weight multiplier for priority classes.")
    parser.add_argument("--priority-oversample", action="store_true", help="Oversample priority classes in the train loader.")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to warm-start every experiment from.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device.")
    args = parser.parse_args()

    output_root = Path(args.output_root) / args.experiment
    output_root.mkdir(parents=True, exist_ok=True)

    base = TrainConfig(
        data_root=args.data_root,
        output_dir=str(output_root),
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
        augment=args.augmentation_policy != "none",
        augmentation_policy=args.augmentation_policy,
        auto_crop=args.auto_crop,
        priority_classes=tuple(args.priority_classes),
        priority_weight=args.priority_weight,
        priority_oversample=args.priority_oversample,
        init_checkpoint=args.init_checkpoint,
        seed=args.seed,
        device=args.device,
    )
    summary_rows: List[Dict[str, object]] = []

    for index, override in enumerate(EXPERIMENT_PRESETS[args.experiment], start=1):
        config_dict = base.__dict__.copy()
        config_dict.update(override)

        if "augmentation_policy" in override:
            config_dict["augment"] = override["augmentation_policy"] != "none"

        run_name = "_".join(f"{key}-{value}" for key, value in override.items())
        config_dict["output_dir"] = str(output_root / f"{index:02d}_{run_name}")
        config = TrainConfig(**config_dict)
        print(f"Running experiment: {run_name}")
        result = train_model(config, output_dir=Path(config.output_dir))
        summary_rows.append(
            {
                "run_name": run_name,
                "override": override,
                "metrics": result["metrics"],
                "class_names": result["class_names"],
            }
        )

    save_summary(output_root, args.experiment, summary_rows)
    print(f"Experiment family finished: {args.experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


