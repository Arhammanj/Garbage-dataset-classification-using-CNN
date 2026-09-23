"""Training entrypoint, shared by all three model variants.

Usage:
    python -m src.train --model baseline --epochs 15
    python -m src.train --model improved --epochs 20
    python -m src.train --model resnet --epochs 10
"""

import argparse
import csv

import torch
import torch.nn as nn

from src.config import MODELS_DIR, TrainConfig
from src.data.dataset import get_dataloaders
from src.models.baseline_cnn import build_model as build_baseline
from src.models.improved_cnn import build_model as build_improved
from src.models.resnet_transfer import build_model as build_resnet

MODEL_BUILDERS = {
    "baseline": build_baseline,
    "improved": build_improved,
    "resnet": build_resnet,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(cfg: TrainConfig) -> str:
    if cfg.model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name '{cfg.model_name}', expected one of {list(MODEL_BUILDERS)}")

    device = get_device()
    print(f"Training '{cfg.model_name}' on {device}")

    train_loader, val_loader, _, classes = get_dataloaders(cfg)
    model = MODEL_BUILDERS[cfg.model_name](len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{cfg.model_name}_best.pt"
    log_path = MODELS_DIR / f"{cfg.model_name}_log.csv"

    # -1 sentinel (not 0.0): guarantees the first epoch always saves a
    # checkpoint, even if its val accuracy happens to be exactly 0.0.
    best_val_acc = -1.0
    log_rows = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"epoch {epoch:>3}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model_name,
                "classes": classes,
                "data_dir": str(cfg.data_dir),
                "epoch": epoch,
                "val_acc": val_acc,
            }, checkpoint_path)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Best val acc: {best_val_acc:.4f} -> saved to {checkpoint_path}")
    return str(checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS), dest="model_name")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    if args.data_dir:
        cfg.data_dir = args.data_dir
    train(cfg)
