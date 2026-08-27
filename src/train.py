"""
Training script for CIFAR-10 image classification.

Reads hyperparameters from a YAML config file, trains a CNN model,
logs metrics in JSON-lines format, and saves checkpoints with early stopping.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Add src directory to path for imports when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    """Load training configuration from a YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        loader: Training data loader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to train on (cpu/cuda).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate the model on the validation set.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    """Main training entry point."""
    # Look for config in container path first, then local path
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    if not config_path.exists():
        print(json.dumps({"event": "error", "message": "Config file not found"}),
              flush=True)
        sys.exit(1)

    config = load_config(str(config_path))

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "setup", "device": str(device)}), flush=True)

    # Model
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    print(json.dumps({
        "event": "model_loaded",
        "architecture": config["model"]["architecture"],
        "num_classes": config["model"]["num_classes"],
    }), flush=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 2),
        max_train_samples=config["data"].get("max_train_samples"),
        max_val_samples=config["data"].get("max_val_samples"),
    )
    print(json.dumps({
        "event": "data_loaded",
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
    }), flush=True)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)

        # Checkpoint saving with early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
            }, save_path)
            print(json.dumps({
                "event": "checkpoint_saved",
                "path": str(save_path),
            }), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch + 1,
                }), flush=True)
                break

    print(json.dumps({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
    }), flush=True)


if __name__ == "__main__":
    main()
