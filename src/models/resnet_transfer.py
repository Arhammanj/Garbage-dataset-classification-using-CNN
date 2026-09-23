"""Transfer learning baseline: ImageNet-pretrained ResNet18, frozen backbone,
new classifier head sized to num_classes.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, freeze_backbone: bool = True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replacing model.fc creates a fresh Linear layer whose params always
    # require grad, even with the rest of the backbone frozen above.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
