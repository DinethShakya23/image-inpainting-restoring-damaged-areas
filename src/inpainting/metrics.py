"""Image-quality metrics for evaluating restorations."""
from __future__ import annotations

from typing import Optional

import numpy as np


def psnr(original: np.ndarray, restored: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher is better).

    Both arrays are expected to share the same shape and value range.
    """
    mse = float(np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def mean_psnr(originals: np.ndarray, restored: np.ndarray, data_range: float = 1.0) -> float:
    """Average PSNR over a batch of images."""
    return float(np.mean([psnr(o, r, data_range) for o, r in zip(originals, restored)]))


def masked_psnr(
    original: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """PSNR computed *only* over the damaged region.

    This is the standard way to score inpainting: it measures how well the model
    reconstructed the missing pixels, without rewarding or penalising it for the
    untouched parts of the image. ``mask`` is a boolean array of shape ``(H, W)``
    (or ``(H, W, C)``) marking the damaged pixels.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2 and original.ndim == 3:
        mask = np.repeat(mask[:, :, None], original.shape[2], axis=2)
    if not mask.any():
        raise ValueError("mask selects no pixels.")

    diff = original.astype(np.float64)[mask] - restored.astype(np.float64)[mask]
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)
