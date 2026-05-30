"""Dataset loading and preprocessing helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

from .config import IMG_SIZE

PathLike = Union[str, Path]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def load_image(path: PathLike, img_size: int = IMG_SIZE) -> np.ndarray:
    """Load a single image as a normalised ``(img_size, img_size, 3)`` array."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def load_dataset(
    dataset_dir: PathLike,
    img_size: int = IMG_SIZE,
    limit: int | None = None,
) -> np.ndarray:
    """Load and normalise every image in ``dataset_dir``.

    Corrupt or unreadable files are skipped with a warning. Returns an array of
    shape ``(N, img_size, img_size, 3)`` with values in ``[0, 1]``.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")

    files = sorted(
        p for p in dataset_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if limit is not None:
        files = files[:limit]

    images = []
    for path in files:
        try:
            images.append(load_image(path, img_size=img_size))
        except Exception as exc:  # noqa: BLE001 - report and skip bad files
            print(f"Skipping unreadable file: {path.name} - {exc}")

    if not images:
        raise ValueError(f"No readable images found in {dataset_dir}")

    print(f"Loaded {len(images)} images from {dataset_dir}")
    return np.stack(images)


def train_test_split_images(
    images: np.ndarray,
    damaged: np.ndarray,
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split paired (clean, damaged) images into train/test sets.

    Returns ``(train_clean, test_clean, train_damaged, test_damaged)``.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        images, damaged, test_size=test_split, random_state=seed
    )
