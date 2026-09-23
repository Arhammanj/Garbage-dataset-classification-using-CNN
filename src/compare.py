"""Runs evaluate.py over the baseline/improved/resnet checkpoints and prints
a side-by-side comparison table.
"""

from src.config import MODELS_DIR
from src.evaluate import evaluate

MODEL_NAMES = ["baseline", "improved", "resnet"]


def main():
    results = []
    for name in MODEL_NAMES:
        checkpoint_path = MODELS_DIR / f"{name}_best.pt"
        if not checkpoint_path.exists():
            print(f"skipping '{name}': no checkpoint at {checkpoint_path} (train it first)")
            continue
        results.append(evaluate(str(checkpoint_path)))

    if not results:
        print("No checkpoints found. Run src/train.py for at least one model first.")
        return

    print(f"\n{'model':<12}{'test accuracy':>15}{'macro f1':>12}")
    for result in results:
        macro_f1 = result["report"]["macro avg"]["f1-score"]
        print(f"{result['model_name']:<12}{result['accuracy']:>15.4f}{macro_f1:>12.4f}")


if __name__ == "__main__":
    main()
