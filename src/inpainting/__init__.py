"""Image inpainting: restoring damaged areas with a CNN autoencoder."""
from __future__ import annotations

__version__ = "1.0.0"

from .data import load_dataset, load_image
from .masking import add_square_mask, damage_batch
from .metrics import mean_psnr, psnr
from .model import build_inpainting_model, compile_model

__all__ = [
    "load_dataset",
    "load_image",
    "add_square_mask",
    "damage_batch",
    "build_inpainting_model",
    "compile_model",
    "psnr",
    "mean_psnr",
]
