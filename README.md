# Image Inpainting — Restoring Damaged Areas

A deep-learning project that **reconstructs missing regions of an image** using a
convolutional encoder–decoder (autoencoder). The network is trained on clean
photographs paired with synthetically *damaged* copies, learning to fill in the
masked-out areas with plausible content.

<p align="center">
  <img src="assets/damage_example.png" alt="Original vs. damaged image" width="640">
</p>

> Originally built as a university image-processing mini-project and refactored
> into a clean, reusable Python package.

---

## How it works

| Stage | Description |
|-------|-------------|
| **Data**     | Images are loaded, resized to `128×128`, and normalised to `[0, 1]`. |
| **Damage**   | A random `32×32` square is zeroed out to simulate a missing region. |
| **Model**    | A CNN autoencoder learns the mapping *damaged → original*. |
| **Training** | Adam optimiser, mean-squared-error loss. |
| **Inference**| A damaged image is passed through the model to recover the masked area. |

### Architecture

```
Input (128×128×3)
  → Conv 64  → MaxPool
  → Conv 128 → MaxPool
  → Conv 256                (bottleneck)
  → ConvT 128 → UpSample
  → ConvT 64  → UpSample
  → Conv 3 (sigmoid)        → Output (128×128×3)
```

---

## Project structure

```
image-inpainting-restoring-damaged-areas/
├── src/inpainting/          # Reusable package
│   ├── config.py            # Paths & hyperparameters
│   ├── data.py              # Loading / preprocessing
│   ├── masking.py           # Synthetic damage
│   ├── model.py             # Encoder-decoder architecture
│   ├── train.py             # Training pipeline (CLI)
│   ├── predict.py           # Inference (CLI)
│   ├── metrics.py           # PSNR
│   └── visualize.py         # Plotting helpers
├── notebooks/demo.ipynb     # End-to-end walkthrough
├── models/                  # Trained model (inpainting_model.h5)
├── assets/                  # Sample image & figures
├── data/                    # Place training images here (git-ignored)
├── outputs/                 # Generated results (git-ignored)
├── pyproject.toml
└── requirements.txt
```

---

## Getting started

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd image-inpainting-restoring-damaged-areas

# 2. Create an environment and install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .          # or: pip install -r requirements.txt
```

> Requires Python 3.9–3.11 (TensorFlow 2.12–2.15).

### Restore an image (uses the bundled trained model)

```bash
python -m inpainting.predict --image assets/sample.jpeg --out outputs/restored.png
```

This damages a fresh image with a random mask and saves an
*original / damaged / restored* comparison. To restore an image that is
**already** damaged, add `--no-mask`.

### Train on your own dataset

```bash
python -m inpainting.train --dataset path/to/images --epochs 10
```

The trained model is written to `models/inpainting_model.h5`.

### Or explore interactively

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Use as a library

```python
import numpy as np
from inpainting.data import load_image
from inpainting.masking import add_square_mask
from inpainting.predict import load_inpainting_model, restore
from inpainting.metrics import psnr

original = load_image("assets/sample.jpeg")
damaged, _ = add_square_mask(original)

model = load_inpainting_model("models/inpainting_model.h5")
restored = restore(model, damaged)

print("PSNR:", round(psnr(original, restored), 2), "dB")
```

---

## Possible improvements

- Train on a larger, more diverse dataset for sharper reconstructions
- Support **irregular / free-form masks** instead of fixed squares
- Add **SSIM / perceptual loss** or a GAN discriminator for realism
- Wrap inference in a small **Gradio / Streamlit** demo

---

## License

Released under the [MIT License](LICENSE).
