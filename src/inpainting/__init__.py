"""Image inpainting: restoring damaged areas with a CNN autoencoder."""
from __future__ import annotations

__version__ = "1.0.0"

from .data import load_dataset, load_image
from .masking import (
    add_square_mask,
    circle_mask,
    damage_batch,
    ellipse_mask,
    random_damage,
    rectangle_mask,
    square_mask,
)
from .metrics import masked_psnr, mean_psnr, psnr
from .model import build_inpainting_model, compile_model

__all__ = [
    "load_dataset",
    "load_image",
    "add_square_mask",
    "square_mask",
    "rectangle_mask",
    "circle_mask",
    "ellipse_mask",
    "random_damage",
    "damage_batch",
    "build_inpainting_model",
    "compile_model",
    "psnr",
    "mean_psnr",
    "masked_psnr",
]
