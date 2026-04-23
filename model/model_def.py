from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from model.lego_cnn import LegoCNN, create_lego_cnn_from_config
from model.stage3_config import DEFAULT_STAGE3_CHECKPOINT, DEFAULT_STAGE3_DATA_ROOT
from model.training_utils import infer_checkpoint_class_names, infer_class_names


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LoadedLegoClassifier:
    checkpoint_path: Path
    data_root: Path
    model: LegoCNN
    config: Dict[str, Any]
    class_names: List[str]
    mean: List[float]
    std: List[float]
    device: str


def load_lego_classifier(
    checkpoint_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    class_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> LoadedLegoClassifier:
    resolved_device = resolve_runtime_device(device)
    checkpoint_path = Path(checkpoint_path or DEFAULT_STAGE3_CHECKPOINT)
    data_root = Path(data_root or DEFAULT_STAGE3_DATA_ROOT)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config = dict(checkpoint["config"])
    candidate_class_names = list(class_names or infer_class_names(data_root))
    resolved_class_names = infer_checkpoint_class_names(checkpoint, default_class_names=candidate_class_names)

    model = create_lego_cnn_from_config(config, num_classes=len(resolved_class_names)).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedLegoClassifier(
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        model=model,
        config=config,
        class_names=resolved_class_names,
        mean=list(checkpoint["mean"]),
        std=list(checkpoint["std"]),
        device=resolved_device,
    )


def resolve_runtime_device(device: Optional[str] = None) -> str:
    if device not in (None, "", "auto"):
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"
