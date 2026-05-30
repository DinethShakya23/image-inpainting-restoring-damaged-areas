"""Image-quality metrics for evaluating restorations."""
from __future__ import annotations

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
