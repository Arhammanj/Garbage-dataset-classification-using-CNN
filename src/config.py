"""Project-wide constants and shared training configuration."""

from dataclasses import dataclass
from pathlib import Path

# Repo root: two levels up from this file (src/config.py -> src -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local default. On Kaggle, override via TrainConfig(data_dir=...) to point at
# /kaggle/input/<dataset-slug>/... (see notebooks/kaggle_train.ipynb).
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = 224  # matches ResNet18's expected input; baseline/improved CNNs
                # use AdaptiveAvgPool2d so they accept the same size.

# ImageNet normalization stats. Required for the pretrained ResNet backbone;
# reused for the from-scratch models too so all three share one transform
# pipeline shape.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    model_name: str  # "baseline" | "improved" | "resnet"
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 2
    data_dir: Path = DATA_DIR
    img_size: int = IMG_SIZE
