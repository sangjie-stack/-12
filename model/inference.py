from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from model.model_def import DEFAULT_STAGE3_CHECKPOINT, DEFAULT_STAGE3_DATA_ROOT, LoadedLegoClassifier, load_lego_classifier
from model.training_utils import build_transforms


@dataclass
class ProbabilityScore:
    class_name: str
    probability: float


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    image_size: tuple[int, int]
    top_probabilities: List[ProbabilityScore]


@lru_cache(maxsize=2)
def load_default_classifier(device: Optional[str] = None) -> LoadedLegoClassifier:
    return load_lego_classifier(
        checkpoint_path=DEFAULT_STAGE3_CHECKPOINT,
        data_root=DEFAULT_STAGE3_DATA_ROOT,
        device=device,
    )


def build_inference_transform(classifier: LoadedLegoClassifier):
    return build_transforms(
        image_size=classifier.config["image_size"],
        mean=classifier.mean,
        std=classifier.std,
        augment=False,
        augmentation_policy="none",
        auto_crop=classifier.config.get("auto_crop", False),
    )


def predict_pil_image(
    image: Image.Image,
    classifier: Optional[LoadedLegoClassifier] = None,
    top_k: int = 5,
) -> PredictionResult:
    classifier = classifier or load_default_classifier()
    transform = build_inference_transform(classifier)
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(torch.device(classifier.device))

    with torch.no_grad():
        probs = torch.softmax(classifier.model(x), dim=1)[0].cpu()

    pairs = sorted(
        zip(classifier.class_names, probs.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_limit = min(len(pairs), max(1, top_k))

    return PredictionResult(
        predicted_class=pairs[0][0],
        confidence=float(pairs[0][1]),
        image_size=image.size,
        top_probabilities=[
            ProbabilityScore(class_name=name, probability=float(prob))
            for name, prob in pairs[:top_limit]
        ],
    )


def predict_image_file(
    image_path: Path,
    classifier: Optional[LoadedLegoClassifier] = None,
    top_k: int = 5,
) -> PredictionResult:
    with Image.open(image_path) as image:
        return predict_pil_image(image, classifier=classifier, top_k=top_k)
