from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRICK_STAGE3_CHECKPOINT = Path(
    "runs/stage2/v6_priority_train_real_round2_recover_1x2_1x4_20260422/best_model.pth"
)
BRICK_STAGE3_DATA_ROOT = Path("data/splits_stage2_v6_priority_train_real_round2_all5")
BRICK_PLATE_STAGE3_CHECKPOINT = Path("runs/stage2/brick_plate_joint_v1/best_model.pth")
BRICK_PLATE_STAGE3_DATA_ROOT = Path("data/splits_brick_plate")


def _exists(path: Path) -> bool:
    return path.exists() if path.is_absolute() else (ROOT / path).exists()


def has_joint_stage3_assets() -> bool:
    return _exists(BRICK_PLATE_STAGE3_CHECKPOINT) and _exists(BRICK_PLATE_STAGE3_DATA_ROOT)


def resolve_default_stage3_mode() -> str:
    return "brick_plate" if has_joint_stage3_assets() else "brick"


DEFAULT_STAGE3_CHECKPOINT = BRICK_PLATE_STAGE3_CHECKPOINT if has_joint_stage3_assets() else BRICK_STAGE3_CHECKPOINT
DEFAULT_STAGE3_DATA_ROOT = BRICK_PLATE_STAGE3_DATA_ROOT if has_joint_stage3_assets() else BRICK_STAGE3_DATA_ROOT
