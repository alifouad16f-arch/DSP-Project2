"""
reorganise_dataset.py
Standalone CLI script to reorganise the raw Mini-ImageNet splits.

Usage:
    python scripts/reorganise_dataset.py \
        --src /path/to/mini_imagenet \
        --dst /path/to/mini_imagenet_fixed
"""

import argparse
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import reorganise_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganise raw Mini-ImageNet train/val/test splits into a "
                    "clean 500-train / 100-test layout."
    )
    parser.add_argument("--src", required=True,
                        help="Path to the extracted raw Mini-ImageNet directory "
                             "(must contain train/, val/, test/ sub-folders).")
    parser.add_argument("--dst", required=True,
                        help="Destination directory for the reorganised dataset.")
    parser.add_argument("--train_per_class", type=int, default=500)
    parser.add_argument("--test_per_class",  type=int, default=100)
    args = parser.parse_args()

    reorganise_dataset(
        src=args.src,
        dst=args.dst,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
