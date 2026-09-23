"""CNN from scratch: plain Conv2d -> ReLU -> MaxPool2d blocks, no
BatchNorm/Dropout. No pretrained weights. This is the control model that
improved_cnn.py and resnet_transfer.py are compared against.
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # AdaptiveAvgPool2d fixes the feature map to 4x4 regardless of input
        # resolution, so the FC layer size below never has to be recomputed
        # by hand if IMG_SIZE changes.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int):
    return BaselineCNN(num_classes)
