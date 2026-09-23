"""Dataset loading: ImageFolder-based, split into train/val/test, wired to
the transforms in src/data/transforms.py.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.config import TrainConfig
from src.data.transforms import eval_transform, get_train_transform


def get_dataloaders(cfg: TrainConfig):
    """Returns (train_loader, val_loader, test_loader, classes).

    classes is the sorted list of class names ImageFolder inferred from the
    dataset's subfolder names — this is the single source of truth for
    class order/count, used to size model output heads.
    """
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "On Kaggle, set TrainConfig(data_dir=Path('/kaggle/input/<dataset-slug>/...'))."
        )

    # Two ImageFolder instances over the same directory: one with augmented
    # transforms (for training), one without (for val/test). ImageFolder
    # lists files in the same sorted order each time, so indices line up
    # between the two — we split indices once and reuse them for both.
    train_view = ImageFolder(data_dir, transform=get_train_transform(cfg.model_name, cfg.img_size))
    eval_view = ImageFolder(data_dir, transform=eval_transform(cfg.img_size))
    classes = train_view.classes

    n = len(train_view)
    if n == 0:
        raise ValueError(f"No images found under {data_dir}. Check the folder layout.")

    n_test = int(n * cfg.test_split)
    n_val = int(n * cfg.val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(n, generator=generator).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_loader = DataLoader(
        Subset(train_view, train_idx), batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_view, val_idx), batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        Subset(eval_view, test_idx), batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
    )

    return train_loader, val_loader, test_loader, classes
