"""
evaluate.py
Evaluation utilities: overall accuracy, per-class degradation analysis,
and compression visualisation.
"""

import torch
import matplotlib.pyplot as plt

from compression import compress_batch, compress_image_wavelet


# ---------------------------------------------------------------------------
# Overall accuracy on a test loader
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module,
             loader,
             device: torch.device = None,
             compress_ratio: float = None) -> float:
    """
    Compute top-1 accuracy on `loader`.

    Args:
        model:          Trained PyTorch model (eval mode set internally).
        loader:         DataLoader for the evaluation set.
        device:         torch.device; auto-detected if None.
        compress_ratio: If provided, images are wavelet-compressed at this ratio
                        before being passed to the model.

    Returns:
        Accuracy as a percentage (0–100).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            if compress_ratio is not None:
                inputs = compress_batch(inputs, compress_ratio)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    label = f"(compressed {compress_ratio}:1)" if compress_ratio else "(uncompressed)"
    print(f"Test Accuracy {label}: {acc:.2f}%")
    return acc


# ---------------------------------------------------------------------------
# Per-class degradation analysis
# ---------------------------------------------------------------------------

def analyze_class_degradation(model: torch.nn.Module,
                               test_loader,
                               class_names: list,
                               ratio: float = 10,
                               device: torch.device = None,
                               top_k: int = 5) -> dict:
    """
    Compare per-class accuracy between uncompressed and wavelet-compressed inputs.

    Args:
        model:        Trained PyTorch model.
        test_loader:  DataLoader for the test set.
        class_names:  List of class name strings.
        ratio:        Wavelet compression ratio.
        device:       torch.device; auto-detected if None.
        top_k:        Number of most/least affected classes to print.

    Returns:
        Dict mapping class name -> {'base_acc', 'comp_acc', 'drop'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Per-class degradation at {ratio}:1 compression...")
    model.eval()

    correct_base = {c: 0 for c in class_names}
    correct_comp = {c: 0 for c in class_names}
    total_class  = {c: 0 for c in class_names}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            preds_base = torch.max(model(inputs), 1)[1]
            comp_inputs = compress_batch(inputs.cpu(), ratio).to(device)
            preds_comp  = torch.max(model(comp_inputs), 1)[1]

            for lbl, pb, pc in zip(labels, preds_base, preds_comp):
                cn = class_names[lbl.item()]
                total_class[cn] += 1
                if pb == lbl:
                    correct_base[cn] += 1
                if pc == lbl:
                    correct_comp[cn] += 1

    class_drops = {}
    for cn in class_names:
        if total_class[cn] > 0:
            base = 100 * correct_base[cn] / total_class[cn]
            comp = 100 * correct_comp[cn] / total_class[cn]
            class_drops[cn] = {"base_acc": base, "comp_acc": comp, "drop": base - comp}

    sorted_drops = sorted(class_drops.items(), key=lambda x: x[1]["drop"], reverse=True)

    print(f"\n--- TOP {top_k} MOST AFFECTED CLASSES ---")
    for cn, stats in sorted_drops[:top_k]:
        print(f"  {cn:15s} | Drop: {stats['drop']:5.1f}%  "
              f"({stats['base_acc']:.1f}% -> {stats['comp_acc']:.1f}%)")

    print(f"\n--- TOP {top_k} MOST ROBUST CLASSES ---")
    for cn, stats in sorted_drops[-top_k:]:
        print(f"  {cn:15s} | Drop: {stats['drop']:5.1f}%  "
              f"({stats['base_acc']:.1f}% -> {stats['comp_acc']:.1f}%)")

    return class_drops


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_compression(image_tensor: torch.Tensor,
                           ratios: list = (2, 5, 10),
                           save_path: str = None) -> None:
    """
    Display an original image alongside its wavelet-compressed versions.

    Args:
        image_tensor: Single image tensor [C, H, W] (normalised).
        ratios:       Compression ratios to display.
        save_path:    If given, saves the figure to this path instead of showing.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def unnorm(t):
        return torch.clamp(t * std + mean, 0, 1)

    cols = [image_tensor] + [compress_image_wavelet(image_tensor, r) for r in ratios]
    titles = ["Original"] + [f"{r}:1 Ratio" for r in ratios]

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, img, title in zip(axes, cols, titles):
        ax.imshow(unnorm(img.cpu()).permute(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
