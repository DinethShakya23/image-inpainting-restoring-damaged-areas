"""Utilities for simulating damage on images.

The model is trained to reconstruct the original image from a *damaged* copy,
where damage is a square region of the image that has been zeroed out. These
helpers create that synthetic damage in a reproducible way.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import MASK_SIZE


def add_square_mask(
    image: np.ndarray,
    mask_size: int = MASK_SIZE,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return a copy of ``image`` with a random black square removed.

    Parameters
    ----------
    image:
        Float image array of shape ``(H, W, C)`` with values in ``[0, 1]``.
    mask_size:
        Side length of the square mask, in pixels.
    rng:
        Optional NumPy random generator for reproducibility. If ``None`` a
        fresh default generator is used.

    Returns
    -------
    damaged:
        Copy of ``image`` with the masked region set to ``0``.
    top_left:
        ``(y, x)`` coordinate of the mask's top-left corner.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    if mask_size >= h or mask_size >= w:
        raise ValueError(
            f"mask_size ({mask_size}) must be smaller than the image ({h}x{w})."
        )

    y = int(rng.integers(0, h - mask_size))
    x = int(rng.integers(0, w - mask_size))

    damaged = image.copy()
    damaged[y : y + mask_size, x : x + mask_size, :] = 0.0
    return damaged, (y, x)


def damage_batch(
    images: np.ndarray,
    mask_size: int = MASK_SIZE,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply :func:`add_square_mask` to every image in a batch."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [add_square_mask(img, mask_size=mask_size, rng=rng)[0] for img in images]
    )
