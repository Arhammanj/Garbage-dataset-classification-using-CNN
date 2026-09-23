"""Loads a checkpoint, computes accuracy/precision/recall/F1, confusion
matrix, and classification report on the held-out test split.
"""

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import TrainConfig
from src.data.dataset import get_dataloaders
from src.train import MODEL_BUILDERS, get_device


def evaluate(checkpoint_path: str, data_dir: str | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint["model_name"]
    classes = checkpoint["classes"]

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name '{model_name}' in checkpoint, expected one of {list(MODEL_BUILDERS)}")

    device = get_device()
    model = MODEL_BUILDERS[model_name](len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Use the same data_dir the checkpoint was trained with (stored at save
    # time) unless the caller explicitly overrides it — otherwise the test
    # split wouldn't match what the model was actually trained/validated on.
    resolved_data_dir = data_dir or checkpoint.get("data_dir")
    cfg = TrainConfig(model_name=model_name, data_dir=resolved_data_dir) if resolved_data_dir else TrainConfig(model_name=model_name)
    _, _, test_loader, loaded_classes = get_dataloaders(cfg)
    if loaded_classes != classes:
        raise ValueError(
            "Class list from the current dataset does not match the checkpoint's "
            f"classes.\ncheckpoint: {classes}\ncurrent data: {loaded_classes}"
        )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=classes)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "model_name": model_name,
        "accuracy": report["accuracy"],
        "report": report,
        "report_str": report_str,
        "confusion_matrix": cm,
        "classes": classes,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.checkpoint, data_dir=args.data_dir)
    print(f"model: {result['model_name']}")
    print(f"test accuracy: {result['accuracy']:.4f}")
    print(result["report_str"])
    print("confusion matrix (rows=true, cols=pred):")
    print(result["confusion_matrix"])
