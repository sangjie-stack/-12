from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from PIL import Image

from model.model_def import DEFAULT_STAGE3_CHECKPOINT, DEFAULT_STAGE3_DATA_ROOT, LoadedLegoClassifier, load_lego_classifier
from model.training_utils import build_transforms

SHADOW_TTA_BLEND_RAW = 0.5
SHADOW_TTA_PARAMS = {
    "shadow_reduce_probability": 1.0,
    "shadow_brightness_floor": 180,
    "shadow_neutral_threshold": 20,
    "shadow_protect_threshold": 200,
    "shadow_strength": 1.0,
}


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


@lru_cache(maxsize=8)
def _build_cached_inference_transform(
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    auto_crop: bool,
    shadow_reduce_probability: float,
    shadow_brightness_floor: int,
    shadow_neutral_threshold: int,
    shadow_protect_threshold: int,
    shadow_strength: float,
):
    return build_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        augment=False,
        augmentation_policy="none",
        auto_crop=auto_crop,
        shadow_reduce_probability=shadow_reduce_probability,
        shadow_brightness_floor=shadow_brightness_floor,
        shadow_neutral_threshold=shadow_neutral_threshold,
        shadow_protect_threshold=shadow_protect_threshold,
        shadow_strength=shadow_strength,
    )


def build_inference_transform(
    classifier: LoadedLegoClassifier,
    use_shadow_reduction: bool = False,
):
    shadow_params = SHADOW_TTA_PARAMS if use_shadow_reduction else {}
    return _build_cached_inference_transform(
        image_size=int(classifier.config["image_size"]),
        mean=tuple(float(item) for item in classifier.mean),
        std=tuple(float(item) for item in classifier.std),
        auto_crop=bool(classifier.config.get("auto_crop", False)),
        shadow_reduce_probability=float(shadow_params.get("shadow_reduce_probability", 0.0)),
        shadow_brightness_floor=int(shadow_params.get("shadow_brightness_floor", 150)),
        shadow_neutral_threshold=int(shadow_params.get("shadow_neutral_threshold", 20)),
        shadow_protect_threshold=int(shadow_params.get("shadow_protect_threshold", 235)),
        shadow_strength=float(shadow_params.get("shadow_strength", 0.85)),
    )


def _predict_probabilities(
    batch: torch.Tensor,
    classifier: LoadedLegoClassifier,
) -> torch.Tensor:
    batch = batch.to(torch.device(classifier.device), non_blocking=True)
    with torch.inference_mode():
        logits = classifier.model(batch)
        return torch.softmax(logits, dim=1).cpu()


def _build_prediction_result(
    probabilities: torch.Tensor,
    classifier: LoadedLegoClassifier,
    image_size: tuple[int, int],
    top_k: int,
) -> PredictionResult:
    top_limit = min(len(classifier.class_names), max(1, top_k))
    top_values, top_indices = torch.topk(probabilities, k=top_limit)
    return PredictionResult(
        predicted_class=classifier.class_names[int(top_indices[0])],
        confidence=float(top_values[0]),
        image_size=image_size,
        top_probabilities=[
            ProbabilityScore(
                class_name=classifier.class_names[int(class_index)],
                probability=float(probability),
            )
            for probability, class_index in zip(top_values.tolist(), top_indices.tolist())
        ],
    )


def predict_pil_image(
    image: Image.Image,
    classifier: Optional[LoadedLegoClassifier] = None,
    top_k: int = 5,
    use_shadow_tta: bool = False,
) -> PredictionResult:
    classifier = classifier or load_default_classifier()
    image = image.convert("RGB")
    if use_shadow_tta:
        raw_transform = build_inference_transform(classifier, use_shadow_reduction=False)
        shadow_transform = build_inference_transform(classifier, use_shadow_reduction=True)
        batch = torch.stack([raw_transform(image), shadow_transform(image)], dim=0)
        probabilities = _predict_probabilities(batch, classifier)
        raw_weight = float(SHADOW_TTA_BLEND_RAW)
        probabilities = probabilities[0] * raw_weight + probabilities[1] * (1.0 - raw_weight)
    else:
        transform = build_inference_transform(classifier, use_shadow_reduction=False)
        batch = transform(image).unsqueeze(0)
        probabilities = _predict_probabilities(batch, classifier)[0]
    return _build_prediction_result(probabilities, classifier, image.size, top_k)


def predict_pil_images(
    images: Sequence[Image.Image],
    classifier: Optional[LoadedLegoClassifier] = None,
    top_k: int = 5,
    use_shadow_tta: bool = False,
) -> List[PredictionResult]:
    if not images:
        return []

    classifier = classifier or load_default_classifier()
    prepared_images = [image.convert("RGB") for image in images]
    if use_shadow_tta:
        raw_transform = build_inference_transform(classifier, use_shadow_reduction=False)
        shadow_transform = build_inference_transform(classifier, use_shadow_reduction=True)
        batch = torch.stack(
            [
                transform(image)
                for image in prepared_images
                for transform in (raw_transform, shadow_transform)
            ],
            dim=0,
        )
        all_probabilities = _predict_probabilities(batch, classifier)
        raw_weight = float(SHADOW_TTA_BLEND_RAW)
        probabilities = torch.stack(
            [
                all_probabilities[index * 2] * raw_weight + all_probabilities[index * 2 + 1] * (1.0 - raw_weight)
                for index in range(len(prepared_images))
            ],
            dim=0,
        )
    else:
        transform = build_inference_transform(classifier, use_shadow_reduction=False)
        batch = torch.stack([transform(image) for image in prepared_images], dim=0)
        probabilities = _predict_probabilities(batch, classifier)
    return [
        _build_prediction_result(prob_row, classifier, image.size, top_k)
        for image, prob_row in zip(prepared_images, probabilities)
    ]


def predict_image_file(
    image_path: Path,
    classifier: Optional[LoadedLegoClassifier] = None,
    top_k: int = 5,
    use_shadow_tta: bool = False,
) -> PredictionResult:
    with Image.open(image_path) as image:
        return predict_pil_image(image, classifier=classifier, top_k=top_k, use_shadow_tta=use_shadow_tta)
