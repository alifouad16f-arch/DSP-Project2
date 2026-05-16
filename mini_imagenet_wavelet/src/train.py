"""
train.py
Training and validation loop for Mini-ImageNet classification.
"""

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

from dataset import get_loaders


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int = 100) -> nn.Module:
    """
    Return a pre-trained model with its classifier head replaced for
    `num_classes` output logits.

    Args:
        name:        'resnet18' or 'mobilenet'.
        num_classes: Number of output classes.
    """
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'resnet18' or 'mobilenet'.")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model: nn.Module,
          train_loader,
          val_loader,
          num_epochs: int = 5,
          lr: float = 1e-4,
          save_path: str = "best_model.pth",
          device: torch.device = None) -> nn.Module:
    """
    Fine-tune `model` and save the checkpoint with the best validation accuracy.

    Args:
        model:        PyTorch model (already adapted for target classes).
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        num_epochs:   Number of training epochs.
        lr:           Adam learning rate.
        save_path:    Path where the best checkpoint is written.
        device:       torch.device; auto-detected if None.

    Returns:
        The model loaded with the best checkpoint weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    print(f"Training on {device} for {num_epochs} epoch(s)...")

    for epoch in range(num_epochs):
        t0 = time.time()

        # --- Training ---
        model.train()
        running_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # --- Validation ---
        model.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | {elapsed:.0f}s | "
            f"Train Loss: {running_loss / len(train_loader):.4f}  "
            f"Train Acc: {train_acc:.2f}%  |  Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path}")

    print("Training complete.")
    model.load_state_dict(torch.load(save_path))
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on Mini-ImageNet")
    parser.add_argument("--model", default="resnet18",
                        choices=["resnet18", "mobilenet"])
    parser.add_argument("--data_dir", required=True,
                        help="Root of the reorganised Mini-ImageNet dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    save_path = args.save_path or f"models/best_{args.model}_baseline.pth"

    train_loader, val_loader, _, _ = get_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    model = build_model(args.model)
    train(model, train_loader, val_loader,
          num_epochs=args.epochs, lr=args.lr, save_path=save_path)
