import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.inference import load_default_classifier, predict_image_file
from model.model_def import DEFAULT_STAGE3_CHECKPOINT, DEFAULT_STAGE3_DATA_ROOT


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_example_predictions(sample_count_per_class: int) -> List[Dict[str, object]]:
    classifier = load_default_classifier()
    examples: List[Dict[str, object]] = []
    test_root = ROOT / DEFAULT_STAGE3_DATA_ROOT / "test"

    for class_name in classifier.class_names:
        class_dir = test_root / class_name
        if not class_dir.exists():
            continue

        sample_paths = [path for path in sorted(class_dir.iterdir()) if path.is_file()][:sample_count_per_class]
        for sample_path in sample_paths:
            prediction = predict_image_file(sample_path, classifier=classifier, top_k=3)
            examples.append(
                {
                    "path": str(sample_path),
                    "expected_class": class_name,
                    "predicted_class": prediction.predicted_class,
                    "confidence": prediction.confidence,
                    "top_probabilities": [
                        {
                            "class_name": item.class_name,
                            "probability": item.probability,
                        }
                        for item in prediction.top_probabilities
                    ],
                }
            )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify stage-3 application prerequisites and sample inference output.")
    parser.add_argument(
        "--sample-count-per-class",
        type=int,
        default=1,
        help="How many sample images to run for each class in the stage-2 test split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the verification summary as JSON.",
    )
    args = parser.parse_args()

    quality_report = load_json(ROOT / "data/quality_report.json")
    stage1_split_report = load_json(ROOT / "data/splits/split_report.json")
    stage2_split_report = load_json(ROOT / DEFAULT_STAGE3_DATA_ROOT / "split_report.json")

    checkpoint_dir = ROOT / DEFAULT_STAGE3_CHECKPOINT.parent
    stage2_metrics = load_json(checkpoint_dir / "test_metrics_rechecked.json")
    if not stage2_metrics:
        stage2_metrics = load_json(checkpoint_dir / "test_metrics.json")

    classifier = load_default_classifier()
    predictions = collect_example_predictions(sample_count_per_class=max(1, args.sample_count_per_class))

    summary = {
        "stage1": {
            "class_count": quality_report.get("class_count", 0),
            "total_images": quality_report.get("dimension_summary", {}).get("total_images", 0),
            "per_class_counts": quality_report.get("per_class_counts", {}),
            "split_report": stage1_split_report,
        },
        "stage2": {
            "checkpoint": str(ROOT / DEFAULT_STAGE3_CHECKPOINT),
            "data_root": str(ROOT / DEFAULT_STAGE3_DATA_ROOT),
            "device": classifier.device,
            "class_names": classifier.class_names,
            "metrics": stage2_metrics,
            "split_report": stage2_split_report,
        },
        "stage3": {
            "sample_predictions": predictions,
        },
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
