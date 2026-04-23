import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lego_cnn import create_lego_cnn_from_config
from model.training_utils import infer_checkpoint_class_names
from utils.dataset_config import VALID_IMAGE_SUFFIXES
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


def iter_photo_images(data_root: Path) -> List[Path]:
    images: List[Path] = []
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                continue
            if not path.name.startswith(f"photo_{class_dir.name}_"):
                continue
            images.append(path)
    return images


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-predict newly added LEGO photo images.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Raw data root containing class folders.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/stage2/photo_batch_eval_7class_v6"),
        help="Directory for JSON and Markdown summaries.",
    )
    parser.add_argument("--class-names", nargs="+", default=None, help="Optional explicit class order for older checkpoints.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    candidate_class_names = args.class_names
    if candidate_class_names is None:
        candidate_class_names = [path.name for path in sorted(args.data_root.iterdir()) if path.is_dir()]
    class_names = infer_checkpoint_class_names(checkpoint, default_class_names=candidate_class_names)
    model = create_lego_cnn_from_config(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform(config, checkpoint["mean"], checkpoint["std"])
    images = iter_photo_images(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, object]] = []
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_class_total = Counter()
    per_class_correct = Counter()

    for image_path in images:
        true_label = image_path.parent.name
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred_index = int(torch.argmax(probs).item())
        pred_label = class_names[pred_index]
        confidence = float(probs[pred_index].item())

        top_probs = sorted(zip(class_names, probs.tolist()), key=lambda item: item[1], reverse=True)
        sample_rows.append(
            {
                "path": str(image_path),
                "filename": image_path.name,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": pred_label == true_label,
                "confidence": confidence,
                "top3": [{"label": label, "prob": prob} for label, prob in top_probs[:3]],
            }
        )
        confusion[true_label][pred_label] += 1
        per_class_total[true_label] += 1
        if pred_label == true_label:
            per_class_correct[true_label] += 1

    overall_total = len(sample_rows)
    overall_correct = sum(1 for row in sample_rows if row["correct"])
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0

    summary_class_names = sorted(set(class_names) | set(per_class_total.keys()))

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_accuracy,
        "per_class": {
            class_name: {
                "total": per_class_total[class_name],
                "correct": per_class_correct[class_name],
                "accuracy": (per_class_correct[class_name] / per_class_total[class_name]) if per_class_total[class_name] else 0.0,
                "pred_distribution": dict(confusion[class_name]),
            }
            for class_name in summary_class_names
        },
        "samples": sample_rows,
    }

    (args.output_dir / "photo_predictions.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class_rows: List[List[str]] = []
    for class_name in summary_class_names:
        total = per_class_total[class_name]
        correct = per_class_correct[class_name]
        accuracy = (correct / total) if total else 0.0
        top_pred = "-"
        if confusion[class_name]:
            top_pred = confusion[class_name].most_common(1)[0][0]
        class_rows.append([class_name, str(total), str(correct), f"{accuracy:.4f}", top_pred])

    wrong_rows: List[List[str]] = []
    for row in sample_rows:
        if row["correct"]:
            continue
        top3 = ", ".join(f"{item['label']}={item['prob']:.3f}" for item in row["top3"])
        wrong_rows.append([row["filename"], row["true_label"], row["pred_label"], f"{row['confidence']:.4f}", top3])

    report_lines = [
        "# Photo Batch Evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Data root: `{args.data_root}`",
        f"- Total photo images: `{overall_total}`",
        f"- Correct predictions: `{overall_correct}`",
        f"- Overall accuracy: `{overall_accuracy:.4f}`",
        "",
        "## Per-class Summary",
        "",
        markdown_table(["Class", "Total", "Correct", "Accuracy", "Most common prediction"], class_rows),
        "",
        "## Misclassified Samples",
        "",
    ]

    if wrong_rows:
        report_lines.append(markdown_table(["Filename", "True", "Pred", "Confidence", "Top3"], wrong_rows))
    else:
        report_lines.append("No misclassified samples.")

    (args.output_dir / "photo_predictions_summary.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"Total photo images: {overall_total}")
    print(f"Correct predictions: {overall_correct}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"JSON saved to: {args.output_dir / 'photo_predictions.json'}")
    print(f"Markdown saved to: {args.output_dir / 'photo_predictions_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
