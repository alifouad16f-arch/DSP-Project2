"""
dataset.py
Dataset loading and split-reorganisation utilities for Mini-ImageNet.
"""

import os
import glob
import shutil

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Standard transform (ImageNet normalisation, resize to 96×96)
# ---------------------------------------------------------------------------

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Reorganise raw Mini-ImageNet splits
# ---------------------------------------------------------------------------

def reorganise_dataset(src: str, dst: str,
                        train_per_class: int = 500,
                        test_per_class: int = 100) -> None:
    """
    Merge the raw train/val/test splits of Mini-ImageNet and redistribute
    into a clean two-split layout:  dst/train  and  dst/test.

    The raw dataset ships as three separate folders (train, val, test) each
    with 100 class sub-directories.  This function pools all 600 images per
    class and writes the first `train_per_class` to dst/train/<class> and
    the remaining `test_per_class` to dst/test/<class>.

    Args:
        src:              Path to the extracted raw dataset (contains train/val/test).
        dst:              Destination root for the reorganised dataset.
        train_per_class:  Images allocated to training per class (default 500).
        test_per_class:   Images allocated to testing per class (default 100).
    """
    new_train = os.path.join(dst, "train")
    new_test = os.path.join(dst, "test")
    os.makedirs(new_train, exist_ok=True)
    os.makedirs(new_test, exist_ok=True)

    # Collect all class-level paths across all three splits
    all_class_paths = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(src, split)
        if not os.path.exists(split_dir):
            continue
        for d in os.listdir(split_dir):
            full = os.path.join(split_dir, d)
            if os.path.isdir(full):
                all_class_paths.append(full)

    print(f"Found {len(all_class_paths)} class-split directories. Reorganising...")

    for class_path in all_class_paths:
        class_name = os.path.basename(class_path)
        os.makedirs(os.path.join(new_train, class_name), exist_ok=True)
        os.makedirs(os.path.join(new_test, class_name), exist_ok=True)

        images = sorted(glob.glob(os.path.join(class_path, "*.*")))
        for img in images[:train_per_class]:
            shutil.copy(img, os.path.join(new_train, class_name, os.path.basename(img)))
        for img in images[train_per_class: train_per_class + test_per_class]:
            shutil.copy(img, os.path.join(new_test, class_name, os.path.basename(img)))

    print(f"Done. Reorganised dataset written to: {dst}")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_loaders(data_dir: str,
                batch_size: int = 64,
                val_fraction: float = 0.2,
                num_workers: int = 2,
                transform=None):
    """
    Build train / val / test DataLoaders for the reorganised Mini-ImageNet.

    Args:
        data_dir:       Root that contains  data_dir/train  and  data_dir/test.
        batch_size:     Batch size for all loaders.
        val_fraction:   Fraction of the training set held out for validation.
        num_workers:    DataLoader worker processes.
        transform:      torchvision transform; defaults to DEFAULT_TRANSFORM.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
    """
    if transform is None:
        transform = DEFAULT_TRANSFORM

    full_train = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_size = int((1 - val_fraction) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print(f"Loaded {len(train_dataset)} train  |  "
          f"{len(val_dataset)} val  |  {len(test_dataset)} test images.")

    return train_loader, val_loader, test_loader, full_train.classes
