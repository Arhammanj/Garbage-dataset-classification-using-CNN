"""Shared train/eval transform pipelines.

Kept separate from dataset.py so the FastAPI backend (Phase 2) can reuse the
exact eval-time transform used during training.
"""

from torchvision import transforms

from src.config import IMG_SIZE, NORM_MEAN, NORM_STD


def eval_transform(img_size: int = IMG_SIZE):
    """Deterministic pipeline: same output every time. Used for val/test
    and later for live inference, so results are reproducible."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def get_train_transform(model_name: str, img_size: int = IMG_SIZE):
    """Training pipeline. The baseline model gets no augmentation (it's the
    control — plain conv net, plain data). improved/resnet get random flips,
    rotation, and color jitter so the model sees a different variant of each
    image every epoch, which fights overfitting on a small dataset."""
    if model_name == "baseline":
        return eval_transform(img_size)

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
