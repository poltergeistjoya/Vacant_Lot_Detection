"""
Generate prediction comparison figures for sample BBLs.

Runs the three figure types from plot_prediction_comparison.py using:
  - UNet kahan_031 (stride 256, threshold 0.425)
  - DeepLabV3+ kahan_027 (stride 512, threshold 0.298)

BBLs are sampled from the Bronx test split (BoroCode 2).

Usage:
  uv run python scripts/run_prediction_comparisons.py
"""
from __future__ import annotations

from pathlib import Path

from vacant_lot.config import _get_shared_root
from vacant_lot.plotting import save_figure
from scripts.plot_prediction_comparison import (
    plot_bbl_inspection,
    plot_model_comparison,
    plot_error_gallery,
)

# ── Run configs ──────────────────────────────────────────────────────────────

UNET_RUN = "kahan_031"
UNET_STRIDE = 256
UNET_THRESH = 0.425

DEEPLAB_RUN = "kahan_027"
DEEPLAB_STRIDE = 512
DEEPLAB_THRESH = 0.298

# ── Sample BBLs (Bronx test split, area > 200 m^2) ──────────────────────────

SAMPLE_BBLS_TEST = [
    2034910001,   # ~1920 m^2, large vacant lot
    2043400095,   # ~1871 m^2, large vacant lot
    2027680034,   # ~1780 m^2, large vacant lot
    2028720218,   # ~444 m^2, medium vacant lot
    2023430004,   # ~438 m^2, medium vacant lot
    2026230187,   # ~341 m^2, smaller vacant lot
]


def main():
    shared_root = _get_shared_root()
    out_dir = shared_root / "outputs" / "figures" / "prediction_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Single-BBL inspections (one per model per BBL) ────────────────────
    for bbl in SAMPLE_BBLS_TEST:
        print(f"\n{'='*60}")
        print(f"BBL {bbl}")
        print(f"{'='*60}")

        # UNet inspection
        fig = plot_bbl_inspection(
            bbl=bbl, arch="unet", run_id=UNET_RUN, split="test",
            stride=UNET_STRIDE, threshold=UNET_THRESH, pad=50,
        )
        save_figure(fig, out_dir / f"bbl_{bbl}_unet_{UNET_RUN}_test.png")

        # DeepLabV3+ inspection
        fig = plot_bbl_inspection(
            bbl=bbl, arch="deeplabv3plus", run_id=DEEPLAB_RUN, split="test",
            stride=DEEPLAB_STRIDE, threshold=DEEPLAB_THRESH, pad=50,
        )
        save_figure(fig, out_dir / f"bbl_{bbl}_deeplab_{DEEPLAB_RUN}_test.png")

    # ── 2. Model comparisons (UNet vs DeepLabV3+ side by side) ───────────────
    for bbl in SAMPLE_BBLS_TEST:
        print(f"\nCompare: BBL {bbl}")
        fig = plot_model_comparison(
            bbl=bbl,
            unet_run=UNET_RUN, deeplab_run=DEEPLAB_RUN,
            split="test", pad=50,
            unet_stride=UNET_STRIDE, unet_threshold=UNET_THRESH,
            deeplab_stride=DEEPLAB_STRIDE, deeplab_threshold=DEEPLAB_THRESH,
        )
        save_figure(fig, out_dir / f"compare_{bbl}_test.png")

    # ── 3. Error galleries (one per model) ───────────────────────────────────
    for arch, run_id, stride, threshold in [
        ("unet", UNET_RUN, UNET_STRIDE, UNET_THRESH),
        ("deeplabv3plus", DEEPLAB_RUN, DEEPLAB_STRIDE, DEEPLAB_THRESH),
    ]:
        print(f"\nGallery: {arch} {run_id}")
        fig = plot_error_gallery(
            arch=arch, run_id=run_id, split="test",
            rows=8, stride=stride, threshold=threshold,
            patch_radius=80, seed=42,
        )
        save_figure(fig, out_dir / f"gallery_{arch}_{run_id}_test.png")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
