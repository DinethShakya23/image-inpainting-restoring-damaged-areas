"""Train the inpainting autoencoder.

Example
-------
    python -m inpainting.train --dataset path/to/images --epochs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import (
    BATCH_SIZE,
    DEFAULT_MODEL_PATH,
    EPOCHS,
    IMG_SIZE,
    MASK_SIZE,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
)
from .data import load_dataset, train_test_split_images
from .masking import damage_batch
from .model import build_inpainting_model, compile_model


def train(
    dataset_dir: str,
    model_out: Path = DEFAULT_MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    mask_size: int = MASK_SIZE,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
    limit: int | None = None,
):
    """Run the full training pipeline and persist the trained model."""
    images = load_dataset(dataset_dir, img_size=img_size, limit=limit)
    damaged = damage_batch(images, mask_size=mask_size, seed=seed)

    train_clean, test_clean, train_damaged, test_damaged = train_test_split_images(
        images, damaged, test_split=test_split, seed=seed
    )

    model = compile_model(build_inpainting_model(img_size=img_size))
    model.summary()

    history = model.fit(
        x=train_damaged,
        y=train_clean,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_damaged, test_clean),
    )

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    print(f"Saved trained model to {model_out}")

    # Persist a small test set so results can be reproduced later.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUTS_DIR / "test_clean.npy", test_clean)
    np.save(OUTPUTS_DIR / "test_damaged.npy", test_damaged)

    return model, history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image inpainting model.")
    parser.add_argument("--dataset", required=True, help="Directory of training images.")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--mask-size", type=int, default=MASK_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of images.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        dataset_dir=args.dataset,
        model_out=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        mask_size=args.mask_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
