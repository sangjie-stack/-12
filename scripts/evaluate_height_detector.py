import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.lego_height_detector import detect_lego_height
from utils.dataset_config import VALID_IMAGE_SUFFIXES


def parse_layer_label(name: str) -> int:
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        raise ValueError(f"Could not parse layer label from directory name: {name}")
    return int(digits)


def iter_labeled_images(data_root: Path) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        true_layer = parse_layer_label(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                samples.append(
                    {
                        "path": image_path,
                        "true_layer": true_layer,
                    }
                )
    return samples


def evaluate_height_dataset(data_root: Path) -> Dict[str, object]:
    samples = iter_labeled_images(data_root)
    if not samples:
        raise ValueError(f"No labeled images found under: {data_root}")

    total_correct = 0
    per_layer_correct: Dict[int, int] = {}
    per_layer_total: Dict[int, int] = {}
    predictions: List[Dict[str, object]] = []

    for sample in samples:
        image_path = sample["path"]
        true_layer = int(sample["true_layer"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = detect_lego_height(image)
        predicted_layer = int(result.height)
        is_correct = predicted_layer == true_layer

        per_layer_total[true_layer] = per_layer_total.get(true_layer, 0) + 1
        per_layer_correct[true_layer] = per_layer_correct.get(true_layer, 0) + int(is_correct)
        total_correct += int(is_correct)

        predictions.append(
            {
                "image": str(image_path),
                "true_layer": true_layer,
                "predicted_layer": predicted_layer,
                "correct": is_correct,
                "confidence": round(float(result.confidence), 4),
                "message": result.message,
            }
        )

    ordered_layers = sorted(per_layer_total)
    per_layer_accuracy = {
        str(layer): per_layer_correct.get(layer, 0) / per_layer_total[layer]
        for layer in ordered_layers
    }
    per_layer_correct_json = {str(layer): per_layer_correct.get(layer, 0) for layer in ordered_layers}
    per_layer_total_json = {str(layer): per_layer_total[layer] for layer in ordered_layers}

    return {
        "sample_count": len(samples),
        "overall_accuracy": total_correct / len(samples),
        "per_layer_accuracy": per_layer_accuracy,
        "per_layer_correct": per_layer_correct_json,
        "per_layer_total": per_layer_total_json,
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LEGO layer-height detection on a labeled folder dataset.")
    parser.add_argument(
        "data_root",
        type=Path,
        help="Folder containing one subfolder per true layer label, for example data/height_eval/1 and data/height_eval/2.",
    )
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save full metrics as JSON.")
    args = parser.parse_args()

    metrics = evaluate_height_dataset(args.data_root)
    print(json.dumps({k: v for k, v in metrics.items() if k != "predictions"}, ensure_ascii=False, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
