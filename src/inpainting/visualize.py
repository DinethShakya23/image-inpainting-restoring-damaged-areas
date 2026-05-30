"""Plotting helpers for inspecting data and results."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def show_comparison(
    original: np.ndarray,
    damaged: np.ndarray,
    restored: np.ndarray,
    save_path: Optional[str] = None,
):
    """Display original / damaged / restored side by side."""
    import matplotlib.pyplot as plt

    titles = ["Original", "Damaged", "Restored"]
    images = [original, damaged, restored]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(np.squeeze(img), 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def show_grid(images: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None):
    """Display a row of images."""
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(np.clip(np.squeeze(images[i]), 0, 1))
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_history(history, save_path: Optional[str] = None):
    """Plot training/validation loss curves from a Keras ``History``."""
    import matplotlib.pyplot as plt

    hist = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["loss"], label="train loss")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
