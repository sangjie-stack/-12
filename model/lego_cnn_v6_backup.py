from pathlib import Path
from typing import List, Optional

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "runs" / "stage2" / "improve_raw128_autocrop_lr5e4_7class_v6" / "best_model.pth"
CLASS_NAMES: List[str] = ["1x1", "1x2", "1x3", "1x4", "2x2", "2x3", "2x4"]
INPUT_SIZE = 128
BASE_CHANNELS = 48
DEPTH = 4
DROPOUT = 0.0
NUM_CLASSES = len(CLASS_NAMES)


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


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


class LegoCNNV6Backup(nn.Module):
    """
    Standalone backup architecture for:
    runs/stage2/improve_raw128_autocrop_lr5e4_7class_v6/best_model.pth
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        in_channels = 3
        layers = []
        for stage_index in range(DEPTH):
            out_channels = BASE_CHANNELS * (2 ** stage_index)
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def create_model(device: Optional[str] = None) -> LegoCNNV6Backup:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return LegoCNNV6Backup().to(resolved_device)


def load_checkpoint(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: Optional[str] = None,
) -> LegoCNNV6Backup:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = create_model(device=resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_checkpoint(device="cpu")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        output = model(dummy)
    print("Loaded backup model successfully.")
    print("Output shape:", tuple(output.shape))
    print("Classes:", CLASS_NAMES)
