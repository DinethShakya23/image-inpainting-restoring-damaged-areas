"""Restore a damaged image with a trained inpainting model.

Example
-------
    # Damage a fresh image with a random square, then restore it:
    python -m inpainting.predict --image photo.jpg --out outputs/restored.png

    # Restore an already-damaged image (no extra mask applied):
    python -m inpainting.predict --image damaged.png --no-mask
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import DEFAULT_MODEL_PATH, IMG_SIZE, MASK_SIZE
from .data import load_image
from .masking import add_square_mask


def load_inpainting_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load a trained Keras model from disk."""
    from tensorflow.keras.models import load_model

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)


def restore(model, damaged: np.ndarray) -> np.ndarray:
    """Run a single damaged image (H, W, 3) through the model."""
    pred = model.predict(damaged[np.newaxis, ...], verbose=0)[0]
    return np.clip(pred, 0.0, 1.0)


def restore_image(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    apply_mask: bool = True,
    mask_size: int = MASK_SIZE,
    img_size: int = IMG_SIZE,
    seed: int | None = None,
):
    """Load an image, optionally damage it, and restore it.

    Returns ``(original, damaged, restored)`` float arrays in ``[0, 1]``.
    """
    original = load_image(image_path, img_size=img_size)

    if apply_mask:
        rng = np.random.default_rng(seed)
        damaged, _ = add_square_mask(original, mask_size=mask_size, rng=rng)
    else:
        damaged = original

    model = load_inpainting_model(model_path)
    restored = restore(model, damaged)
    return original, damaged, restored


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a damaged image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--out", default=None, help="Where to save the comparison figure.")
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Treat the input as already damaged (do not add a synthetic mask).",
    )
    parser.add_argument("--mask-size", type=int, default=MASK_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    original, damaged, restored = restore_image(
        image_path=args.image,
        model_path=args.model,
        apply_mask=not args.no_mask,
        mask_size=args.mask_size,
        seed=args.seed,
    )

    from .visualize import show_comparison

    out = args.out or "outputs/restoration.png"
    show_comparison(original, damaged, restored, save_path=out)
    print(f"Saved comparison figure to {out}")


if __name__ == "__main__":
    main()
