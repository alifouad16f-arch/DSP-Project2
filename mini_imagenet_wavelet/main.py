"""
main.py
Entry point for the Mini-ImageNet Wavelet Compression study.

Runs all three phases for both ResNet-18 and MobileNetV2:
  Phase 1 – Baseline training on uncompressed data
  Phase 2 – Evaluate baseline models on wavelet-compressed inputs
  Phase 3 – Re-train on pre-compressed data; evaluate robustness

Usage:
    python main.py --data_dir /path/to/mini_imagenet_fixed
    python main.py --data_dir /path/to/mini_imagenet_fixed --epochs 10 --models resnet18
    python main.py --data_dir /path/to/mini_imagenet_fixed --skip_phase3
"""

import argparse
import os
import glob
import shutil

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import get_loaders, DEFAULT_TRANSFORM
from src.compression import compress_image_wavelet, compress_batch
from src.train import build_model, train
from src.evaluate import evaluate, analyze_class_degradation, visualize_compression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPRESSION_RATIOS = [2, 5, 10]
TARGET_RATIO = 10


def make_compressed_train_dir(data_dir: str, ratio: int, device: torch.device) -> str:
    """
    Generate a pre-compressed training set on disk (used for Phase 3).
    Returns the path to the compressed train directory.
    """
    source_dir = os.path.join(data_dir, "train")
    target_dir = os.path.join(f"./mini_imagenet_compressed_{ratio}", "train")

    if os.path.exists(target_dir):
        print(f"Compressed training dir already exists: {target_dir}  (skipping generation)")
        return target_dir

    print(f"Generating {ratio}:1 compressed training set at: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    for class_name in os.listdir(source_dir):
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

    load_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    raw_dataset = ImageFolder(source_dir, transform=load_transform)
    raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)

    for i, (img_tensor, label) in enumerate(raw_loader):
        class_name = raw_dataset.classes[label.item()]
        original_path, _ = raw_dataset.samples[i]
        file_name = os.path.basename(original_path)

        compressed = compress_image_wavelet(img_tensor[0], ratio)
        compressed = torch.clamp(compressed, 0, 1)

        save_image(compressed, os.path.join(target_dir, class_name, file_name))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(raw_dataset)} images...")

    print(f"Compressed training set ready: {target_dir}")
    return target_dir


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase1(model_name: str, train_loader, val_loader, num_epochs: int,
               device: torch.device, lr: float) -> nn.Module:
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — {model_name.upper()} baseline training (uncompressed)")
    print(f"{'='*60}")

    model = build_model(model_name)
    save_path = os.path.join("models", f"best_{model_name}_baseline.pth")
    model = train(model, train_loader, val_loader,
                  num_epochs=num_epochs, lr=lr,
                  save_path=save_path, device=device)
    return model


def run_phase2(model_name: str, model: nn.Module, test_loader,
               class_names: list, device: torch.device) -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — {model_name.upper()} wavelet compression evaluation")
    print(f"{'='*60}")

    # Overall accuracy at each compression ratio
    for ratio in COMPRESSION_RATIOS:
        evaluate(model, test_loader, device=device, compress_ratio=ratio)

    # Per-class degradation at the hardest ratio
    analyze_class_degradation(model, test_loader, class_names,
                               ratio=TARGET_RATIO, device=device)

    # Visual sample
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    vis_path = os.path.join("results", f"{model_name}_compression_visual.png")
    visualize_compression(images[0], ratios=COMPRESSION_RATIOS, save_path=vis_path)


def run_phase3(model_name: str, data_dir: str, val_loader, test_loader,
               num_epochs: int, device: torch.device, lr: float) -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — {model_name.upper()} re-training on compressed data")
    print(f"{'='*60}")

    comp_train_dir = make_compressed_train_dir(data_dir, TARGET_RATIO, device)

    comp_dataset = ImageFolder(comp_train_dir, transform=DEFAULT_TRANSFORM)
    comp_loader = DataLoader(comp_dataset, batch_size=64, shuffle=True, num_workers=2)

    model = build_model(model_name)
    save_path = os.path.join("models", f"best_{model_name}_compressed.pth")
    model = train(model, comp_loader, val_loader,
                  num_epochs=num_epochs, lr=lr,
                  save_path=save_path, device=device)

    print(f"\nEvaluating vaccinated {model_name} on {TARGET_RATIO}:1 compressed test set...")
    evaluate(model, test_loader, device=device, compress_ratio=TARGET_RATIO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mini-ImageNet Wavelet Compression Study — full pipeline"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Root of the reorganised Mini-ImageNet dataset "
                             "(must contain train/ and test/ sub-folders).")
    parser.add_argument("--models", nargs="+",
                        default=["resnet18", "mobilenet"],
                        choices=["resnet18", "mobilenet"],
                        help="Which model(s) to run (default: both).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs for each phase (default: 5).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate (default: 1e-4). "
                             "MobileNet typically benefits from 1e-3.")
    parser.add_argument("--skip_phase3", action="store_true",
                        help="Skip the compression-robustness re-training phase.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data once — shared across models
    train_loader, val_loader, test_loader, class_names = get_loaders(
        args.data_dir, batch_size=args.batch_size
    )

    for model_name in args.models:
        # MobileNet trains better at a slightly higher LR
        lr = 1e-3 if model_name == "mobilenet" and args.lr == 1e-4 else args.lr

        baseline_model = run_phase1(model_name, train_loader, val_loader,
                                    args.epochs, device, lr)

        # Uncompressed baseline accuracy
        print(f"\nBaseline ({model_name}) uncompressed test accuracy:")
        evaluate(baseline_model, test_loader, device=device)

        run_phase2(model_name, baseline_model, test_loader, class_names, device)

        if not args.skip_phase3:
            run_phase3(model_name, args.data_dir, val_loader, test_loader,
                       args.epochs, device, lr)

    print("\nAll phases complete. Weights saved in models/  |  Plots saved in results/")


if __name__ == "__main__":
    main()
