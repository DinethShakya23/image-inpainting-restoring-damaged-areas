"""Build a restoration gallery (original | damaged | restored) from a folder.

Runs every image in a directory through the trained model and stitches the
results into a single PNG, annotated with per-image PSNR.

Example
-------
    python scripts/make_gallery.py --input data --out assets/gallery.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Make the src/ package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inpainting.config import IMG_SIZE  # noqa: E402
from inpainting.data import load_image  # noqa: E402
from inpainting.masking import random_damage, square_mask  # noqa: E402
from inpainting.metrics import masked_psnr  # noqa: E402
from inpainting.predict import load_inpainting_model, restore  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_pil(arr: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
    return img.resize((IMG_SIZE * scale, IMG_SIZE * scale), Image.NEAREST)


def build_gallery(
    input_dir: Path,
    model_path: Path,
    out_path: Path,
    size_range: tuple[int, int] = (12, 28),
    scale: int = 2,
    seed: int = 7,
    random_shapes: bool = True,
) -> None:
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _EXTS)
    if not files:
        raise SystemExit(f"No images found in {input_dir}")

    model = load_inpainting_model(model_path)
    font = _load_font(18)
    header_font = _load_font(20)

    cell = IMG_SIZE * scale
    pad, gap, header_h, row_label_h = 14, 16, 32, 24
    row_h = header_h + cell + row_label_h
    width = pad * 2 + cell * 3 + gap * 2
    height = pad * 2 + row_h * len(files)

    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    rng = np.random.default_rng(seed)
    for r, path in enumerate(files):
        original = load_image(path)
        if random_shapes:
            damaged, mask, shape = random_damage(
                original, size_range=size_range, rng=rng
            )
        else:
            damaged, mask = square_mask(original, size=size_range[1], rng=rng)
            shape = "square"
        restored = restore(model, damaged)

        # Masked-region PSNR is the meaningful inpainting score: it grades only
        # the pixels the model had to reconstruct.
        m_dmg = masked_psnr(original, damaged, mask)
        m_res = masked_psnr(original, restored, mask)

        y0 = pad + r * row_h
        if r == 0:
            for c, title in enumerate(("Original", f"Damaged ({shape})", "Restored")):
                x = pad + c * (cell + gap)
                draw.text((x, y0), title, fill=(20, 20, 20), font=header_font)
        else:
            # Label just the damaged column with this row's shape.
            draw.text(
                (pad + cell + gap, y0), shape, fill=(20, 20, 20), font=header_font
            )

        panels = [_to_pil(a, scale) for a in (original, damaged, restored)]
        for c, panel in enumerate(panels):
            x = pad + c * (cell + gap)
            canvas.paste(panel, (x, y0 + header_h))

        label = (
            f"{path.name}    masked-region PSNR: "
            f"{m_dmg:.1f} dB (damaged) -> {m_res:.1f} dB (restored)"
        )
        draw.text((pad, y0 + header_h + cell + 3), label, fill=(60, 60, 60), font=font)
        print(f"{path.name} [{shape}]: {m_dmg:.2f} -> {m_res:.2f} dB (masked region)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved gallery to {out_path} ({width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a restoration gallery image.")
    parser.add_argument("--input", default="data", help="Folder of source images.")
    parser.add_argument("--model", default="models/inpainting_model.h5")
    parser.add_argument("--out", default="assets/gallery.png")
    parser.add_argument(
        "--min-size", type=int, default=12, help="Smallest mask size (px)."
    )
    parser.add_argument(
        "--max-size", type=int, default=28, help="Largest mask size (px)."
    )
    parser.add_argument(
        "--squares-only",
        action="store_true",
        help="Use fixed squares instead of random shapes.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    build_gallery(
        input_dir=Path(args.input),
        model_path=Path(args.model),
        out_path=Path(args.out),
        size_range=(args.min_size, args.max_size),
        random_shapes=not args.squares_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
