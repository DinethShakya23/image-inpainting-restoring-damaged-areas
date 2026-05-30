"""Convolutional encoder-decoder model for image inpainting."""
from __future__ import annotations

from .config import CHANNELS, IMG_SIZE


def build_inpainting_model(img_size: int = IMG_SIZE, channels: int = CHANNELS):
    """Build and return an (uncompiled) encoder-decoder CNN.

    The network takes a damaged image and learns to reconstruct the original.

    Architecture
    ------------
    Encoder : Conv(64) -> Pool -> Conv(128) -> Pool
    Bottleneck : Conv(256)
    Decoder : ConvT(128) -> Up -> ConvT(64) -> Up -> Conv(3, sigmoid)
    """
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(img_size, img_size, channels))

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(channels, (3, 3), activation="sigmoid", padding="same")(x)

    return models.Model(inputs, outputs, name="inpainting_autoencoder")


def compile_model(model, learning_rate: float = 1e-3):
    """Compile ``model`` with Adam + MSE loss (in-place) and return it."""
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
