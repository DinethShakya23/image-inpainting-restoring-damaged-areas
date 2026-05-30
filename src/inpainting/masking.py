"""Utilities for simulating damage on images.

The model is trained to reconstruct the original image from a *damaged* copy,
where a region of the image has been zeroed out. These helpers create that
synthetic damage — in several shapes, sizes and locations — reproducibly.

Each masking function returns ``(damaged, mask)`` where ``mask`` is a boolean
array of shape ``(H, W)`` marking the damaged pixels.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .config import MASK_SHAPES, MASK_SIZE

Mask = np.ndarray  # boolean (H, W)


def _apply(image: np.ndarray, mask: Mask) -> np.ndarray:
    """Zero out the masked pixels of a copy of ``image``."""
    damaged = image.copy()
    damaged[mask] = 0.0
    return damaged


def _resolve_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def square_mask(
    image: np.ndarray,
    size: int = MASK_SIZE,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Mask]:
    """Damage a random ``size`` x ``size`` square."""
    return rectangle_mask(image, height=size, width=size, rng=rng)


def rectangle_mask(
    image: np.ndarray,
    height: int = MASK_SIZE,
    width: int = MASK_SIZE,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Mask]:
    """Damage a random axis-aligned rectangle of the given size."""
    rng = _resolve_rng(rng)
    h, w = image.shape[:2]
    if height >= h or width >= w:
        raise ValueError(
            f"mask ({height}x{width}) must be smaller than the image ({h}x{w})."
        )

    y = int(rng.integers(0, h - height))
    x = int(rng.integers(0, w - width))

    mask = np.zeros((h, w), dtype=bool)
    mask[y : y + height, x : x + width] = True
    return _apply(image, mask), mask


def ellipse_mask(
    image: np.ndarray,
    radius_y: int = MASK_SIZE // 2,
    radius_x: int = MASK_SIZE // 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Mask]:
    """Damage a random (axis-aligned) ellipse with the given radii."""
    rng = _resolve_rng(rng)
    h, w = image.shape[:2]
    if 2 * radius_y >= h or 2 * radius_x >= w:
        raise ValueError(
            f"ellipse radii ({radius_y},{radius_x}) too large for image ({h}x{w})."
        )

    cy = int(rng.integers(radius_y, h - radius_y))
    cx = int(rng.integers(radius_x, w - radius_x))

    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) / radius_y) ** 2 + ((xx - cx) / radius_x) ** 2 <= 1.0
    return _apply(image, mask), mask


def circle_mask(
    image: np.ndarray,
    radius: int = MASK_SIZE // 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Mask]:
    """Damage a random circle of the given radius."""
    return ellipse_mask(image, radius_y=radius, radius_x=radius, rng=rng)


def random_damage(
    image: np.ndarray,
    shapes: Sequence[str] = MASK_SHAPES,
    size_range: Tuple[int, int] = (16, 32),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Mask, str]:
    """Apply one randomly chosen shape, at a random size and location.

    Parameters
    ----------
    shapes:
        Pool of shape names to choose from (see :data:`config.MASK_SHAPES`).
    size_range:
        Inclusive ``(min, max)`` bound on the mask's size, in pixels. For
        squares/rectangles this is the side length; for circles/ellipses it is
        the diameter.

    Returns
    -------
    ``(damaged, mask, shape_name)``.
    """
    rng = _resolve_rng(rng)
    shape = str(rng.choice(shapes))
    lo, hi = size_range

    def _dim() -> int:
        return int(rng.integers(lo, hi + 1))

    if shape == "square":
        size = _dim()
        damaged, mask = square_mask(image, size=size, rng=rng)
    elif shape == "rectangle":
        damaged, mask = rectangle_mask(image, height=_dim(), width=_dim(), rng=rng)
    elif shape == "circle":
        damaged, mask = circle_mask(image, radius=max(1, _dim() // 2), rng=rng)
    elif shape == "ellipse":
        damaged, mask = ellipse_mask(
            image, radius_y=max(1, _dim() // 2), radius_x=max(1, _dim() // 2), rng=rng
        )
    else:
        raise ValueError(f"Unknown shape: {shape!r}. Choose from {tuple(MASK_SHAPES)}.")

    return damaged, mask, shape


# Backward-compatible helpers -------------------------------------------------
def add_square_mask(
    image: np.ndarray,
    mask_size: int = MASK_SIZE,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Damage a random square (kept for backward compatibility).

    Returns ``(damaged, top_left)`` where ``top_left`` is the ``(y, x)`` corner.
    """
    rng = _resolve_rng(rng)
    damaged, mask = square_mask(image, size=mask_size, rng=rng)
    ys, xs = np.where(mask)
    top_left = (int(ys.min()), int(xs.min()))
    return damaged, top_left


def damage_batch(
    images: np.ndarray,
    mask_size: int = MASK_SIZE,
    seed: Optional[int] = None,
    shape: str = "square",
    size_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Apply damage to every image in a batch.

    With ``shape="square"`` (default) a fixed ``mask_size`` square is used. With
    ``shape="random"`` each image gets a random shape/size/location from
    :func:`random_damage`, which is useful for making training robust to varied
    damage. ``size_range`` overrides the random size bounds.
    """
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        if shape == "random":
            kwargs = {"rng": rng}
            if size_range is not None:
                kwargs["size_range"] = size_range
            out.append(random_damage(img, **kwargs)[0])
        else:
            out.append(square_mask(img, size=mask_size, rng=rng)[0])
    return np.stack(out)
