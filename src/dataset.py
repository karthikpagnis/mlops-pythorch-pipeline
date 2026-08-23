"""
CIFAR-10 dataset loading utilities.

Provides data transforms and DataLoader creation for training and validation.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel-wise mean and std (precomputed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Return data transforms for CIFAR-10.

    Training transforms include data augmentation (random crop, horizontal flip).
    Validation transforms only normalize.

    Args:
        train: If True, apply training augmentations.

    Returns:
        A torchvision Compose transform pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple:
    """
    Create CIFAR-10 train and validation DataLoaders.

    Downloads the dataset if not already present in data_dir.

    Args:
        data_dir: Root directory to store/load CIFAR-10 data.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
