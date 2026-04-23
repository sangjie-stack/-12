from typing import Any, Mapping, Optional

import torch
from torch import nn


MODEL_VARIANTS = ("baseline", "deep_residual")
DEFAULT_MODEL_VARIANT = "baseline"
DEFAULT_CONV_POOL_DEPTH = 4


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return self.act(x)


class LegoCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        base_channels: int = 32,
        depth: int = DEFAULT_CONV_POOL_DEPTH,
        dropout: float = 0.3,
        variant: str = DEFAULT_MODEL_VARIANT,
        blocks_per_stage: int = 1,
        fc_hidden_dim: Optional[int] = None,
        fc_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1")
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported model variant: {variant}")

        self.variant = variant

        if variant == "baseline":
            in_channels = 3
            layers = []
            for stage_index in range(depth):
                out_channels = base_channels * (2 ** stage_index)
                layers.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels
            self.features = nn.Sequential(*layers)
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, num_classes),
            )
        else:
            stem_channels = base_channels
            layers = [
                conv3x3(3, stem_channels),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = stem_channels
            for stage_index in range(depth):
                out_channels = base_channels * (2 ** stage_index)
                stage_layers = []
                for block_index in range(blocks_per_stage):
                    block_in_channels = in_channels if block_index == 0 else out_channels
                    stage_layers.append(ResidualBlock(block_in_channels, out_channels))
                stage_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(nn.Sequential(*stage_layers))
                in_channels = out_channels
            self.features = nn.Sequential(*layers)
            hidden_dim = fc_hidden_dim if fc_hidden_dim is not None else max(128, in_channels // 2)
            classifier_layers = [
                nn.Flatten(),
                nn.Linear(in_channels, hidden_dim),
            ]
            if fc_batch_norm:
                classifier_layers.append(nn.BatchNorm1d(hidden_dim))
            classifier_layers.extend(
                [
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                ]
            )
            classifier = nn.Sequential(*classifier_layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model_kwargs(config: Mapping[str, Any]) -> dict:
    variant = str(config.get("model_variant", DEFAULT_MODEL_VARIANT))
    if variant not in MODEL_VARIANTS:
        variant = DEFAULT_MODEL_VARIANT
    default_blocks = 1 if variant == "baseline" else 2
    fc_hidden_dim = config.get("fc_hidden_dim")
    return {
        "base_channels": int(config.get("base_channels", 32)),
        "depth": int(config.get("depth", DEFAULT_CONV_POOL_DEPTH)),
        "dropout": float(config.get("dropout", 0.3)),
        "variant": variant,
        "blocks_per_stage": int(config.get("blocks_per_stage", default_blocks)),
        "fc_hidden_dim": int(fc_hidden_dim) if fc_hidden_dim is not None else None,
        "fc_batch_norm": bool(config.get("fc_batch_norm", False)),
    }


def create_lego_cnn_from_config(config: Mapping[str, Any], num_classes: int) -> LegoCNN:
    return LegoCNN(
        num_classes=num_classes,
        **build_model_kwargs(config),
    )



