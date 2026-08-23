"""
Model definition for CIFAR-10 image classification.

Provides a factory function to create torchvision models adapted for
CIFAR-10 (32x32 images, 10 classes).
"""

import torch.nn as nn
from torchvision import models


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Create and return a CNN model adapted for CIFAR-10.

    The ResNet-18 architecture is modified for 32x32 inputs:
    - First conv layer uses kernel_size=3, stride=1, padding=1 (instead of 7x7)
    - MaxPool layer is removed (unnecessary for small spatial dims)

    Args:
        architecture: Model architecture name (currently supports 'resnet18').
        num_classes: Number of output classes.

    Returns:
        A PyTorch nn.Module ready for training.
    """
    if architecture == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)

        # Adapt for CIFAR-10 (32x32 images instead of ImageNet 224x224)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.bn1 = nn.BatchNorm2d(64)
        model.maxpool = nn.Identity()  # Remove maxpool for small images

        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}. Supported: ['resnet18']"
    )
