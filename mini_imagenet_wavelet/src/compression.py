"""
compression.py
Haar Wavelet image compression utilities.
"""

import pywt
import numpy as np
import torch


def compress_image_wavelet(image_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Compress a single PyTorch image tensor using a 2D Haar Discrete Wavelet Transform.

    Hard thresholding is applied: the bottom (1 - 1/ratio) fraction of wavelet
    coefficients (by absolute value) is zeroed out, then the image is reconstructed
    via the Inverse DWT.

    Args:
        image_tensor: Float tensor of shape [C, H, W], already on CPU or moved here.
        ratio:        Compression ratio (e.g. 2 keeps 50 % of coefficients,
                      10 keeps 10 %).

    Returns:
        Reconstructed float tensor of shape [C, H, W] on CPU.
    """
    img_np = image_tensor.cpu().numpy()
    C, H, W = img_np.shape
    compressed_img = np.zeros_like(img_np)

    for c in range(C):
        channel = img_np[c]

        coeffs = pywt.dwt2(channel, "haar")
        cA, (cH, cV, cD) = coeffs

        all_coeffs = np.concatenate(
            (cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten())
        )
        total_coeffs = len(all_coeffs)
        kept_coeffs = max(1, int(total_coeffs / ratio))

        sorted_abs = np.sort(np.abs(all_coeffs))
        threshold = sorted_abs[-kept_coeffs]

        cA_t = pywt.threshold(cA, threshold, mode="hard")
        cH_t = pywt.threshold(cH, threshold, mode="hard")
        cV_t = pywt.threshold(cV, threshold, mode="hard")
        cD_t = pywt.threshold(cD, threshold, mode="hard")

        reconstructed = pywt.idwt2((cA_t, (cH_t, cV_t, cD_t)), "haar")
        compressed_img[c] = reconstructed[:H, :W]

    return torch.from_numpy(compressed_img).float()


def compress_batch(batch: torch.Tensor, ratio: float) -> torch.Tensor:
    """Apply compress_image_wavelet to every image in a batch [B, C, H, W]."""
    return torch.stack([compress_image_wavelet(img, ratio) for img in batch])
