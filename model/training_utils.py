import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from model.lego_cnn import DEFAULT_CONV_POOL_DEPTH, DEFAULT_MODEL_VARIANT, LegoCNN
from model.lego_dataset import LegoDataset
from utils.dataset_config import BRICK_CLASSES, list_class_directories
from utils.image_preprocess import auto_crop_to_square, reduce_white_background_shadows

DEFAULT_AUGMENTATION_POLICY = "none"
AUGMENTATION_POLICIES = ("none", "affine", "affine_flip", "affine_color")
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    data_root: str = "data/splits"
    output_dir: str = "runs/stage2/baseline"
    image_size: int = 64
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    optimizer: str = "adam"
    base_channels: int = 32
    depth: int = DEFAULT_CONV_POOL_DEPTH
    model_variant: str = DEFAULT_MODEL_VARIANT
    blocks_per_stage: int = 2
    fc_hidden_dim: int = 256
    fc_batch_norm: bool = True
    patience: int = 5
    num_workers: int = 0
    augment: bool = False
    augmentation_policy: str = DEFAULT_AUGMENTATION_POLICY
    auto_crop: bool = False
    shadow_reduce_probability: float = 0.0
    shadow_brightness_floor: int = 150
    shadow_neutral_threshold: int = 20
    shadow_protect_threshold: int = 235
    shadow_strength: float = 0.85
    priority_classes: Tuple[str, ...] = ()
    priority_weight: float = 1.0
    priority_oversample: bool = False
    init_checkpoint: Optional[str] = None
    seed: int = 42
    device: str = "cpu"


class RandomShadowReduction:
    def __init__(
        self,
        probability: float,
        brightness_floor: int,
        neutral_threshold: int,
        protect_white_threshold: int,
        strength: float,
    ) -> None:
        self.probability = float(max(0.0, min(1.0, probability)))
        self.brightness_floor = brightness_floor
        self.neutral_threshold = neutral_threshold
        self.protect_white_threshold = protect_white_threshold
        self.strength = strength

    def __call__(self, image):
        if self.probability <= 0.0 or random.random() >= self.probability:
            return image
        return reduce_white_background_shadows(
            image,
            brightness_floor=self.brightness_floor,
            neutral_threshold=self.neutral_threshold,
            protect_white_threshold=self.protect_white_threshold,
            strength=self.strength,
        )


class EarlyStopping:
    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_augmentation_policy(
    augment: bool = False,
    augmentation_policy: str = DEFAULT_AUGMENTATION_POLICY,
) -> str:
    policy = augmentation_policy or DEFAULT_AUGMENTATION_POLICY
    if policy not in AUGMENTATION_POLICIES:
        raise ValueError(
            f"Unsupported augmentation policy: {policy}. "
            f"Expected one of {', '.join(AUGMENTATION_POLICIES)}."
        )
    if policy == DEFAULT_AUGMENTATION_POLICY and augment:
        return "affine"
    return policy


def infer_class_names(
    data_root: Path,
    default_class_names: Optional[Sequence[str]] = None,
) -> List[str]:
    candidates = list(default_class_names or BRICK_CLASSES)
    split_roots = [data_root / split_name for split_name in ("train", "val", "test")]

    discovered: List[str] = []
    for split_root in split_roots:
        if not split_root.exists():
            continue
        for class_dir in list_class_directories(split_root):
            class_name = class_dir.name
            if class_name not in discovered:
                discovered.append(class_name)

    if not discovered:
        return candidates

    ordered = [class_name for class_name in candidates if class_name in discovered]
    ordered.extend(class_name for class_name in discovered if class_name not in ordered)
    return ordered


def infer_checkpoint_class_names(
    checkpoint: Dict[str, object],
    default_class_names: Optional[Sequence[str]] = None,
) -> List[str]:
    class_names = checkpoint.get("class_names")
    if class_names:
        return list(class_names)

    model_state = checkpoint.get("model_state_dict", {})
    output_dim = infer_output_dim_from_state_dict(model_state)
    if output_dim is not None:
        if default_class_names is not None:
            default_list = list(default_class_names)
            if len(default_list) == output_dim:
                return default_list
            if len(default_list) < output_dim:
                raise ValueError(
                    f"Provided class_names only contain {len(default_list)} classes, "
                    f"but the checkpoint expects {output_dim} outputs."
                )
        if output_dim == len(BRICK_CLASSES):
            return list(BRICK_CLASSES)
        raise ValueError(
            "Checkpoint does not record class_names, and its output dimension does not match "
            "the current default class list. Please provide the class names explicitly."
        )

    return list(default_class_names or BRICK_CLASSES)


def infer_output_dim_from_state_dict(model_state: Dict[str, object]) -> Optional[int]:
    linear_weights = [
        tensor
        for key, tensor in model_state.items()
        if key.endswith("weight") and getattr(tensor, "ndim", None) == 2
    ]
    if not linear_weights:
        return None
    return int(linear_weights[-1].shape[0])


def build_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    augment: bool = False,
    augmentation_policy: str = DEFAULT_AUGMENTATION_POLICY,
    auto_crop: bool = False,
    shadow_reduce_probability: float = 0.0,
    shadow_brightness_floor: int = 150,
    shadow_neutral_threshold: int = 20,
    shadow_protect_threshold: int = 235,
    shadow_strength: float = 0.85,
):
    transform_list: List[object] = []
    if shadow_reduce_probability > 0.0:
        transform_list.append(
            RandomShadowReduction(
                probability=shadow_reduce_probability,
                brightness_floor=shadow_brightness_floor,
                neutral_threshold=shadow_neutral_threshold,
                protect_white_threshold=shadow_protect_threshold,
                strength=shadow_strength,
            )
        )
    if auto_crop:
        transform_list.append(transforms.Lambda(auto_crop_to_square))
    policy = resolve_augmentation_policy(augment=augment, augmentation_policy=augmentation_policy)
    if policy in {"affine", "affine_flip", "affine_color"}:
        transform_list.append(
            transforms.RandomAffine(
                degrees=18,
                translate=(0.12, 0.12),
                scale=(0.92, 1.08),
                fill=255,
            )
        )
    if policy == "affine_flip":
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    if policy == "affine_color":
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.18,
                contrast=0.18,
                saturation=0.12,
                hue=0.02,
            )
        )
    transform_list.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transform_list)


def compute_mean_std(
    train_root: Path,
    image_size: int,
    class_names: Optional[Sequence[str]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    auto_crop: bool = False,
) -> Tuple[List[float], List[float]]:
    transform_list: List[object] = []
    if auto_crop:
        transform_list.append(transforms.Lambda(auto_crop_to_square))
    transform_list.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = LegoDataset(
        train_root,
        class_names=class_names,
        transform=transforms.Compose(transform_list),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0
    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        total_pixels += batch_pixels

    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sq_sum / total_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: Sequence[float],
    std: Sequence[float],
    augment: bool,
    augmentation_policy: str,
    auto_crop: bool,
    shadow_reduce_probability: float = 0.0,
    shadow_brightness_floor: int = 150,
    shadow_neutral_threshold: int = 20,
    shadow_protect_threshold: int = 235,
    shadow_strength: float = 0.85,
    class_names: Optional[Sequence[str]] = None,
    priority_classes: Optional[Sequence[str]] = None,
    priority_weight: float = 1.0,
    priority_oversample: bool = False,
) -> Dict[str, DataLoader]:
    class_names = list(class_names or BRICK_CLASSES)
    train_dataset = LegoDataset(
        data_root / "train",
        class_names=class_names,
        transform=build_transforms(
            image_size,
            mean,
            std,
            augment=augment,
            augmentation_policy=augmentation_policy,
            auto_crop=auto_crop,
            shadow_reduce_probability=shadow_reduce_probability,
            shadow_brightness_floor=shadow_brightness_floor,
            shadow_neutral_threshold=shadow_neutral_threshold,
            shadow_protect_threshold=shadow_protect_threshold,
            shadow_strength=shadow_strength,
        ),
    )
    eval_transform = build_transforms(
        image_size,
        mean,
        std,
        augment=False,
        augmentation_policy=DEFAULT_AUGMENTATION_POLICY,
        auto_crop=auto_crop,
        shadow_reduce_probability=0.0,
    )
    val_dataset = LegoDataset(data_root / "val", class_names=class_names, transform=eval_transform)
    test_dataset = LegoDataset(data_root / "test", class_names=class_names, transform=eval_transform)

    priority_class_set = {name for name in (priority_classes or []) if name in class_names}
    train_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if priority_oversample and priority_class_set and priority_weight > 1.0:
        sample_weights = []
        for _, label in train_dataset.samples:
            class_name = class_names[label]
            sample_weights.append(priority_weight if class_name in priority_class_set else 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader_kwargs["sampler"] = sampler
    else:
        train_loader_kwargs["shuffle"] = True

    return {
        "train": DataLoader(train_dataset, **train_loader_kwargs),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def build_class_weights(
    class_names: Sequence[str],
    priority_classes: Optional[Sequence[str]] = None,
    priority_weight: float = 1.0,
) -> torch.Tensor:
    weights = torch.ones(len(class_names), dtype=torch.float32)
    if priority_weight <= 1.0:
        return weights

    priority_class_set = set(priority_classes or [])
    for index, class_name in enumerate(class_names):
        if class_name in priority_class_set:
            weights[index] = priority_weight
    return weights


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def create_model(config: TrainConfig, num_classes: int) -> LegoCNN:
    return LegoCNN(
        num_classes=num_classes,
        base_channels=config.base_channels,
        depth=config.depth,
        dropout=config.dropout,
        variant=config.model_variant,
        blocks_per_stage=config.blocks_per_stage,
        fc_hidden_dim=config.fc_hidden_dim,
        fc_batch_norm=config.fc_batch_norm,
    )


def create_optimizer(model: nn.Module, config: TrainConfig):
    name = config.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def load_initial_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_class_names = infer_checkpoint_class_names(checkpoint, default_class_names=class_names)
    if list(checkpoint_class_names) != list(class_names):
        raise ValueError(
            "Initial checkpoint class order does not match the current dataset classes: "
            f"{checkpoint_class_names} vs {list(class_names)}"
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"Failed to load initial checkpoint from {checkpoint_path}. "
            "Please make sure the architecture flags match the checkpoint config."
        ) from exc
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "config": checkpoint.get("config"),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_count += images.size(0)

    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_history(history: Dict[str, List[float]], output_dir: Path, prefix: str = "history") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_accuracy"], label="train_accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_accuracy.png", dpi=150)
    plt.close()


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    model.train(False)
    class_names = list(class_names or BRICK_CLASSES)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    per_class_correct = {class_name: 0 for class_name in class_names}
    per_class_total = {class_name: 0 for class_name in class_names}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_count += images.size(0)

            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                class_name = class_names[label]
                per_class_total[class_name] += 1
                if prediction == label:
                    per_class_correct[class_name] += 1

    per_class_accuracy = {
        class_name: (per_class_correct[class_name] / per_class_total[class_name]) if per_class_total[class_name] else 0.0
        for class_name in class_names
    }

    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
        "per_class_accuracy": per_class_accuracy,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }


def train_model(config: TrainConfig, output_dir: Optional[Path] = None) -> Dict[str, object]:
    set_seed(config.seed)
    output_dir = output_dir or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(config.data_root)
    class_names = infer_class_names(data_root)
    augmentation_policy = resolve_augmentation_policy(
        augment=config.augment,
        augmentation_policy=config.augmentation_policy,
    )
    resolved_config = asdict(config)
    resolved_config["augment"] = augmentation_policy != DEFAULT_AUGMENTATION_POLICY
    resolved_config["augmentation_policy"] = augmentation_policy
    mean, std = compute_mean_std(
        data_root / "train",
        image_size=config.image_size,
        class_names=class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        auto_crop=config.auto_crop,
    )
    save_json(output_dir / "normalize_stats.json", {"mean": mean, "std": std})

    loaders = build_dataloaders(
        data_root=data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mean=mean,
        std=std,
        augment=augmentation_policy != DEFAULT_AUGMENTATION_POLICY,
        augmentation_policy=augmentation_policy,
        auto_crop=config.auto_crop,
        shadow_reduce_probability=config.shadow_reduce_probability,
        shadow_brightness_floor=config.shadow_brightness_floor,
        shadow_neutral_threshold=config.shadow_neutral_threshold,
        shadow_protect_threshold=config.shadow_protect_threshold,
        shadow_strength=config.shadow_strength,
        class_names=class_names,
        priority_classes=config.priority_classes,
        priority_weight=config.priority_weight,
        priority_oversample=config.priority_oversample,
    )

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model = create_model(config, num_classes=len(class_names)).to(device)
    init_checkpoint_info = None
    if config.init_checkpoint:
        init_checkpoint_path = Path(config.init_checkpoint)
        if not init_checkpoint_path.is_absolute():
            init_checkpoint_path = ROOT / init_checkpoint_path
        init_checkpoint_info = load_initial_checkpoint(
            model,
            checkpoint_path=init_checkpoint_path,
            device=device,
            class_names=class_names,
        )
    class_weights = build_class_weights(
        class_names,
        priority_classes=config.priority_classes,
        priority_weight=config.priority_weight,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = create_optimizer(model, config)
    early_stopping = EarlyStopping(patience=config.patience)

    dummy_batch_size = 2 if getattr(config, "fc_batch_norm", False) else 1
    dummy_input = torch.randn(dummy_batch_size, 3, config.image_size, config.image_size, device=device)
    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_input)
    model.train()
    save_json(
        output_dir / "model_summary.json",
        {
            "config": resolved_config,
            "class_names": class_names,
            "num_classes": len(class_names),
            "parameter_count": count_parameters(model),
            "dummy_output_shape": list(dummy_output.shape),
            "initial_checkpoint": init_checkpoint_info,
            "class_weights": {
                class_name: float(weight)
                for class_name, weight in zip(class_names, class_weights.detach().cpu().tolist())
            },
        },
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": resolved_config,
                    "class_names": class_names,
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                output_dir / "best_model.pth",
            )

        if early_stopping.step(val_metrics["loss"]):
            break

    plot_history(history, output_dir)
    save_json(output_dir / "history.json", history)

    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_loader(model, loaders["test"], criterion, device, class_names=class_names)
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "per_class_accuracy": test_metrics["per_class_accuracy"],
        "per_class_correct": test_metrics["per_class_correct"],
        "per_class_total": test_metrics["per_class_total"],
    }
    save_json(output_dir / "test_metrics.json", final_metrics)
    return {
        "class_names": class_names,
        "mean": mean,
        "std": std,
        "history": history,
        "metrics": final_metrics,
    }


