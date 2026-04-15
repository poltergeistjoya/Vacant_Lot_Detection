"""
Thesis architecture diagrams — original figures, no copyright issues.

Produces (PDF for LaTeX + PNG for Google Docs):
  outputs/figures/fig_unet_architecture.{pdf,png}
  outputs/figures/fig_deeplab_variants.{pdf,png}
  outputs/figures/fig_atrous_convolution.{pdf,png}

Usage:
    uv run python scripts/plot_thesis_architectures.py
    uv run python scripts/plot_thesis_architectures.py --only unet
    uv run python scripts/plot_thesis_architectures.py --only deeplab
    uv run python scripts/plot_thesis_architectures.py --only atrous
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "STIX Two Text",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
    }
)

# ── output path ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
SHARED_ROOT = _HERE.parents[2]          # scripts/ → main/ → Vacant_Lot_Detection/
OUT = SHARED_ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette (consistent with repo style) ─────────────────────────────────────
C_ENC   = "#3B78C4"   # encoder blue
C_DEC   = "#2E9B52"   # decoder green
C_BN    = "#7B4FA6"   # bottleneck purple
C_POOL  = "#C0392B"   # max-pool red
C_UP    = "#27AE60"   # up-conv green
C_SKIP  = "#E67E22"   # skip / copy orange
C_ASPP  = "#8E44AD"   # ASPP module purple
C_GRAY  = "#7F8C8D"   # neutral gray
C_LGRAY = "#BDC3C7"   # light gray
C_IMG   = "#D5D8DC"   # image block


# ── low-level drawing helpers ─────────────────────────────────────────────────

def _rect(ax, cx, cy, w, h, fc, ec="white", lw=0.7, alpha=1.0, zorder=2, radius=0.0):
    bs = f"round,pad={radius}" if radius else "square,pad=0"
    p = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=bs,
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def _label(ax, cx, cy, text, color="white", fs=6.5, zorder=3, ha="center", va="center"):
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=fs, color=color, zorder=zorder)


def _arrow(ax, x0, y0, x1, y1, color="#555", lw=1.0, head=7, ls="solid", zorder=3):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            mutation_scale=head,
            linestyle=ls,
        ),
        zorder=zorder,
    )


def _save(fig, name):
    for ext in ("pdf", "png"):
        p = OUT / f"fig_{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"  saved: {p.relative_to(SHARED_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — U-Net (64 base channels)
# ─────────────────────────────────────────────────────────────────────────────

def _unet_feature_block(ax, cx, cy, h, channels, fc, BW=0.30):
    """Two side-by-side conv blocks as vertical rectangles."""
    gap = 0.06
    for dx in (-BW / 2 - gap / 2, BW / 2 + gap / 2):
        _rect(ax, cx + dx, cy, BW, h, fc)
    ax.text(cx, cy + h / 2 + 0.13, str(channels),
            ha="center", va="bottom", fontsize=6, color="#222", zorder=3)


def plot_unet():
    """U-Net encoder-decoder with skip connections, 256×256×4 input."""

    # --- layout constants ---
    # Heights are log2-proportional to spatial size for visual clarity
    # spatial: 256 128  64   32   16
    # height:  3.0 2.4  1.8  1.2  0.7
    HEIGHTS = [3.0, 2.4, 1.8, 1.2, 0.7]
    CHANNELS = [64, 128, 256, 512, 1024]
    N = 4          # number of encoder/decoder levels (excluding bottleneck)
    BW = 0.30      # single block width
    base_y = 0.7   # bottom y of all blocks

    # x centres for each pair of conv blocks
    # Encoder: levels 0-3 going right and down
    # Decoder: mirrored, levels 3-0
    enc_x = [1.1, 2.4, 3.7, 5.0]
    bn_x  = 6.5
    dec_x = [8.0, 9.3, 10.6, 11.9]

    enc_y = [base_y + HEIGHTS[i] / 2 for i in range(N)]
    bn_y  = base_y + HEIGHTS[4] / 2
    dec_y = [base_y + HEIGHTS[3 - i] / 2 for i in range(N)]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(-0.3, 14.2)
    ax.set_ylim(0.0, 7.0)
    ax.axis("off")

    # ── encoder blocks ────────────────────────────────────────────────────────
    for i in range(N):
        _unet_feature_block(ax, enc_x[i], enc_y[i], HEIGHTS[i], CHANNELS[i], C_ENC)

    # ── bottleneck ────────────────────────────────────────────────────────────
    _unet_feature_block(ax, bn_x, bn_y, HEIGHTS[4], CHANNELS[4], C_BN)

    # ── decoder blocks ────────────────────────────────────────────────────────
    for i in range(N):
        _unet_feature_block(ax, dec_x[i], dec_y[i], HEIGHTS[3 - i], CHANNELS[3 - i], C_DEC)

    # ── downsampling arrows (encoder → next level) ────────────────────────────
    for i in range(N - 1):
        _arrow(ax,
               enc_x[i] + BW / 2 + 0.06 / 2 + 0.03,  enc_y[i]  - HEIGHTS[i]   / 2,
               enc_x[i + 1] - BW / 2 - 0.06 / 2 - 0.03, enc_y[i + 1] + HEIGHTS[i + 1] / 2,
               color=C_POOL, lw=1.0, head=6)
    # last encoder → bottleneck
    _arrow(ax,
           enc_x[3] + BW / 2 + 0.06 / 2 + 0.03, enc_y[3] - HEIGHTS[3] / 2,
           bn_x    - BW / 2 - 0.06 / 2 - 0.03, bn_y    + HEIGHTS[4] / 2,
           color=C_POOL, lw=1.0, head=6)

    # ── upsampling arrows (bottleneck → decoder, decoder → next level) ────────
    _arrow(ax,
           bn_x     + BW / 2 + 0.06 / 2 + 0.03, bn_y    + HEIGHTS[4] / 2,
           dec_x[0] - BW / 2 - 0.06 / 2 - 0.03, dec_y[0] - HEIGHTS[3] / 2,
           color=C_UP, lw=1.0, head=6)
    for i in range(N - 1):
        _arrow(ax,
               dec_x[i] + BW / 2 + 0.06 / 2 + 0.03,     dec_y[i]     + HEIGHTS[3 - i]     / 2,
               dec_x[i + 1] - BW / 2 - 0.06 / 2 - 0.03, dec_y[i + 1] - HEIGHTS[3 - i - 1] / 2,
               color=C_UP, lw=1.0, head=6)

    # ── skip connections (copy & concat) ─────────────────────────────────────
    for i in range(N):
        x0 = enc_x[i] + BW / 2 + 0.06 / 2 + 0.03
        x1 = dec_x[N - 1 - i] - BW / 2 - 0.06 / 2 - 0.03
        y  = enc_y[i]
        _arrow(ax, x0, y, x1, y, color=C_SKIP, lw=0.9, head=6, ls=(0, (5, 4)))

    # ── spatial size annotations (below bottom level) ─────────────────────────
    for i, (x, y, sp) in enumerate(zip(enc_x, enc_y, [256, 128, 64, 32])):
        if i == 0:
            ax.text(x, y - HEIGHTS[i] / 2 - 0.15, f"{sp}×{sp}",
                    ha="center", va="top", fontsize=5.5, color="#666", zorder=3)
    ax.text(bn_x, bn_y - HEIGHTS[4] / 2 - 0.15, "16×16",
            ha="center", va="top", fontsize=5.5, color="#666", zorder=3)
    ax.text(dec_x[3], dec_y[3] - HEIGHTS[0] / 2 - 0.15, "256×256",
            ha="center", va="top", fontsize=5.5, color="#666", zorder=3)

    # ── input label ───────────────────────────────────────────────────────────
    _arrow(ax, 0.05, enc_y[0], enc_x[0] - BW / 2 - 0.06 / 2 - 0.05, enc_y[0],
           color="#333", lw=1.2, head=8)
    ax.text(0.0, enc_y[0], "input\n256×256×4",
            ha="right", va="center", fontsize=7, color="#222",
            transform=ax.transData)

    # ── 1×1 conv output block ─────────────────────────────────────────────────
    out_cx = dec_x[3] + 1.0
    out_cy = dec_y[3]
    out_h  = HEIGHTS[0]
    _rect(ax, out_cx, out_cy, 0.28, out_h, C_GRAY)
    _label(ax, out_cx, out_cy, "1×1\nconv", fs=5.5)
    ax.text(out_cx, out_cy + out_h / 2 + 0.13, "2",
            ha="center", va="bottom", fontsize=6, color="#222", zorder=3)
    _arrow(ax, dec_x[3] + BW / 2 + 0.06 / 2 + 0.05, dec_y[3],
           out_cx - 0.14, out_cy, color="#333", lw=1.0, head=7)
    _arrow(ax, out_cx + 0.14, out_cy, out_cx + 0.7, out_cy,
           color="#333", lw=1.2, head=8)
    ax.text(out_cx + 0.75, out_cy, "segmentation\nmap 256×256",
            ha="left", va="center", fontsize=7, color="#222")

    # ── legend ────────────────────────────────────────────────────────────────
    items = [
        (C_ENC,  "encoder conv 3×3, BN, ReLU"),
        (C_DEC,  "decoder conv 3×3, BN, ReLU"),
        (C_BN,   "bottleneck"),
        (C_POOL, "max pool 2×2 (downsample)"),
        (C_UP,   "up-conv 2×2 (upsample)"),
        (C_SKIP, "copy & concatenate (skip)"),
    ]
    lx, ly = 0.1, 6.7
    cols = 3
    for j, (color, label) in enumerate(items):
        col = j % cols
        row = j // cols
        x = lx + col * 4.5
        y = ly - row * 0.38
        ax.add_patch(mpatches.Rectangle((x, y - 0.12), 0.22, 0.24,
                                         fc=color, ec="none", zorder=2))
        ax.text(x + 0.32, y, label, va="center", fontsize=6.5, color="#333")

    ax.set_title("U-Net architecture (64 base channels, 256×256 NAIP input)",
                 fontsize=9, pad=6)

    fig.tight_layout(pad=0.5)
    _save(fig, "unet_architecture")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — DeepLab variants: SPP / Encoder-Decoder / E-D + Atrous
# ─────────────────────────────────────────────────────────────────────────────

def _mini_feat_stack(ax, cx, base_y, n_levels, fc, h0=0.45, gap=0.08, w=1.1):
    """Draw n_levels stacked feature-map blocks shrinking upward."""
    y = base_y
    for k in range(n_levels):
        scale = 1.0 - 0.15 * k
        _rect(ax, cx, y + (h0 * scale) / 2, w * scale, h0 * scale, fc, radius=0.03)
        if k < n_levels - 1:
            y += h0 * scale + gap
    return y + h0 * (1.0 - 0.15 * (n_levels - 1))  # top y


def _feat_box(ax, cx, cy, w, h, fc, label="", fs=6.5, lw=0.7):
    _rect(ax, cx, cy, w, h, fc, ec="white", lw=lw, radius=0.04)
    if label:
        _label(ax, cx, cy, label, fs=fs)


def plot_deeplab_variants():
    """Three-panel figure showing SPP, Encoder-Decoder, and E-D+Atrous."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.88, bottom=0.08)

    panel_data = [
        {
            "title": "(a) Spatial Pyramid Pooling",
            "model": "DeepLabv3",
        },
        {
            "title": "(b) Encoder-Decoder",
            "model": "U-Net",
        },
        {
            "title": "(c) Encoder-Decoder\nwith Atrous Conv",
            "model": "DeepLabv3+",
        },
    ]

    for idx, (ax, info) in enumerate(zip(axes, panel_data)):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 8.5)
        ax.axis("off")

        model_label = info["model"]
        ax.set_title(info["title"], fontsize=8, pad=4)
        ax.text(2.0, -0.05, model_label, ha="center", va="top",
                fontsize=8, color="#333", style="italic")

        CX = 2.0   # horizontal centre of each panel

        # ── image block ───────────────────────────────────────────────────────
        _feat_box(ax, CX, 0.5, 2.0, 0.6, C_IMG, "Image", fs=7)
        _arrow(ax, CX, 0.8, CX, 1.3, color="#555", lw=1.0, head=6)

        if idx == 0:
            # ── (a) Spatial Pyramid Pooling ────────────────────────────────
            # Backbone
            _feat_box(ax, CX, 1.75, 2.0, 0.65, C_ENC, "DCNN Backbone", fs=6.5)
            _arrow(ax, CX, 2.08, CX, 2.55, color="#555", lw=1.0, head=6)
            # Feature map
            _feat_box(ax, CX, 2.85, 2.2, 0.55, C_ENC, "Feature Map\n(output stride 8/16)", fs=5.8)
            # Fan out to 4 ASPP branches
            branches = [0.6, 1.4, 2.6, 3.4]
            for bx in branches:
                _arrow(ax, CX, 3.12, bx, 3.55, color=C_ASPP, lw=0.8, head=5)
            labels = ["r=6", "r=12", "r=18", "pool"]
            for bx, lb in zip(branches, labels):
                _feat_box(ax, bx, 3.85, 0.62, 0.48, C_ASPP, lb, fs=6)
            # Concat / 1×1 conv
            for bx in branches:
                _arrow(ax, bx, 4.09, CX, 4.55, color=C_ASPP, lw=0.8, head=5)
            _feat_box(ax, CX, 4.85, 2.0, 0.55, C_ASPP, "Concat → 1×1 Conv", fs=6)
            _arrow(ax, CX, 5.13, CX, 5.7, color="#555", lw=1.0, head=6)
            # Upsample
            _feat_box(ax, CX, 6.0, 2.0, 0.55, C_DEC, "Bilinear Upsample ×8", fs=6.5)
            _arrow(ax, CX, 6.28, CX, 6.85, color="#555", lw=1.0, head=6)
            _feat_box(ax, CX, 7.15, 2.0, 0.55, "#27AE60", "Prediction", fs=7)

            # Scale indicators on the left
            ax.text(0.1, 0.5,  "×1",   va="center", fontsize=6, color="#888")
            ax.text(0.1, 2.85, "×0.5", va="center", fontsize=6, color="#888")
            ax.text(0.1, 7.15, "×1",   va="center", fontsize=6, color="#888")

        elif idx == 1:
            # ── (b) Encoder-Decoder ────────────────────────────────────────
            EX, DX = 1.2, 2.8   # encoder / decoder x

            # Encoder side (going up)
            enc_ys  = [1.75, 2.65, 3.55, 4.45]
            enc_chs = [64, 128, 256, 512]
            enc_h = 0.55
            for ey, ch in zip(enc_ys, enc_chs):
                _feat_box(ax, EX, ey, 1.0, enc_h, C_ENC, str(ch), fs=6.5)
            for i in range(len(enc_ys) - 1):
                _arrow(ax, EX, enc_ys[i] + enc_h / 2, EX, enc_ys[i + 1] - enc_h / 2,
                       color=C_POOL, lw=0.8, head=5)
            _arrow(ax, EX, enc_ys[-1] + enc_h / 2, 2.0, 5.3,
                   color=C_POOL, lw=0.8, head=5)
            # Bottleneck
            _feat_box(ax, 2.0, 5.6, 1.3, 0.55, C_BN, "1024", fs=6.5)
            _arrow(ax, 2.0, 5.88, DX, 4.72, color=C_UP, lw=0.8, head=5)

            # Decoder side (going down)
            dec_ys  = [4.45, 3.55, 2.65, 1.75]
            dec_chs = [512, 256, 128, 64]
            for dy, ch in zip(dec_ys, dec_chs):
                _feat_box(ax, DX, dy, 1.0, enc_h, C_DEC, str(ch), fs=6.5)
            for i in range(len(dec_ys) - 1):
                _arrow(ax, DX, dec_ys[i] - enc_h / 2, DX, dec_ys[i + 1] + enc_h / 2,
                       color=C_UP, lw=0.8, head=5)

            # Skip connections
            for ey, dy in zip(enc_ys, dec_ys):
                _arrow(ax, EX + 0.5, ey, DX - 0.5, dy,
                       color=C_SKIP, lw=0.8, head=5, ls=(0, (4, 3)))

            # Output
            _arrow(ax, CX, 1.3, CX, 0.8, color="#555", lw=1.0, head=6)
            _arrow(ax, DX, dec_ys[-1] - enc_h / 2, CX, 1.3,
                   color="#555", lw=0.8, head=5)
            _feat_box(ax, CX, 7.15, 2.0, 0.55, "#27AE60", "Prediction", fs=7)
            _arrow(ax, DX, dec_ys[-1] - enc_h / 2 - 0.05, CX, 7.15 - 0.28,
                   color="#555", lw=0.8, head=5)

        else:
            # ── (c) Encoder-Decoder with Atrous Conv (DeepLabv3+) ──────────
            EX, DX = 1.2, 2.8

            # Encoder with atrous convolutions
            enc_ys  = [1.75, 2.65, 3.55]
            enc_labels = ["Stride 2", "Stride 2", "Atrous (r=2)"]
            enc_h = 0.55
            for ey, lb in zip(enc_ys, enc_labels):
                _feat_box(ax, EX, ey, 1.05, enc_h, C_ENC, lb, fs=5.8)
            for i in range(len(enc_ys) - 1):
                _arrow(ax, EX, enc_ys[i] + enc_h / 2, EX, enc_ys[i + 1] - enc_h / 2,
                       color=C_POOL, lw=0.8, head=5)
            _arrow(ax, EX, enc_ys[-1] + enc_h / 2, 2.0, 4.45,
                   color=C_POOL, lw=0.8, head=5)

            # ASPP at top of encoder
            _feat_box(ax, 2.0, 4.75, 1.8, 0.55, C_ASPP, "ASPP", fs=7)
            _arrow(ax, 2.0, 5.03, 2.0, 5.55, color="#555", lw=0.8, head=5)
            _feat_box(ax, 2.0, 5.85, 1.8, 0.55, C_ASPP, "1×1 Conv\n(reduce 256ch)", fs=5.8)
            _arrow(ax, 2.0, 6.13, DX, 3.83, color=C_UP, lw=0.8, head=5)

            # Low-level features from encoder (skip)
            _arrow(ax, EX, enc_ys[0], DX - 0.5, 2.05,
                   color=C_SKIP, lw=0.8, head=5, ls=(0, (4, 3)))
            _feat_box(ax, EX - 0.05, enc_ys[0] - 0.05, 1.05, enc_h, C_ENC,
                      "Low-level\nfeatures", fs=5.5)

            # Decoder
            dec_ys  = [3.55, 2.65, 1.85]
            dec_lbls = ["Concat+Conv", "Conv", "Conv"]
            for dy, lb in zip(dec_ys, dec_lbls):
                _feat_box(ax, DX, dy, 1.1, enc_h, C_DEC, lb, fs=5.8)
            for i in range(len(dec_ys) - 1):
                _arrow(ax, DX, dec_ys[i] - enc_h / 2, DX, dec_ys[i + 1] + enc_h / 2,
                       color=C_UP, lw=0.8, head=5)

            # Output
            _arrow(ax, DX, dec_ys[-1] - enc_h / 2, CX, 7.15 - 0.28,
                   color="#555", lw=0.8, head=5)
            _feat_box(ax, CX, 7.15, 2.0, 0.55, "#27AE60", "Prediction", fs=7)

    fig.suptitle(
        "Segmentation architecture families — SPP, Encoder-Decoder, Encoder-Decoder with Atrous Conv",
        fontsize=9, y=0.97,
    )
    _save(fig, "deeplab_variants")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Atrous (Dilated) Convolution
# ─────────────────────────────────────────────────────────────────────────────

def _draw_atrous_panel(ax, rate, grid_n=9):
    """
    Draw a single atrous-convolution panel on ax.
    grid_n × grid_n feature-map cells; kernel is 3×3 with dilation=rate.
    """
    ax.set_xlim(-0.1, grid_n + 0.1)
    ax.set_ylim(-0.7, grid_n + 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background feature-map grid
    for r in range(grid_n):
        for c in range(grid_n):
            ax.add_patch(mpatches.Rectangle(
                (c, r), 1, 1,
                fc="#EBF5FF", ec="#BBCDE0", lw=0.4, zorder=1,
            ))

    # Determine which cells the 3×3 kernel with dilation=rate touches
    # Centre of the kernel (rounded so all sampled positions fit in the grid)
    margin = rate * 2  # keep sampled cells inside the grid
    centre = grid_n // 2

    kernel_offsets = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1), ( 0, 0), ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)]

    for dr, dc in kernel_offsets:
        cr = centre + dr * rate
        cc = centre + dc * rate
        is_center = (dr == 0 and dc == 0)
        ax.add_patch(mpatches.FancyBboxPatch(
            (cc + 0.06, cr + 0.06), 0.88, 0.88,
            boxstyle="round,pad=0.04",
            fc="#E84040" if is_center else "#E87040",
            ec="white", lw=0.5, zorder=3,
        ))

    # Dashed lines connecting kernel positions (to show spacing)
    if rate > 1:
        for dr, dc in kernel_offsets:
            if (dr, dc) == (0, 0):
                continue
            cr = centre + dr * rate
            cc = centre + dc * rate
            ax.plot(
                [centre + 0.5, cc + 0.5],
                [centre + 0.5, cr + 0.5],
                color="#888", lw=0.5, ls="--", zorder=2,
            )

    # Header box (matches style of original figure)
    header_text = f"Conv\nkernel: 3×3\nrate: {rate}"
    ax.text(grid_n / 2, grid_n + 0.6, header_text,
            ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="#D6EAF8", ec="#2E86C1", lw=0.8))

    # "rate = X" label inside the feature map
    ax.text(grid_n - 0.3, grid_n - 0.3, f"rate = {rate}",
            ha="right", va="top", fontsize=7, color="#555", zorder=4)

    # Feature map label below
    ax.text(grid_n / 2, -0.4, "Feature map",
            ha="center", va="top", fontsize=7.5, color="#333")


def plot_atrous_convolution():
    """Three panels: rate=1, rate=6, rate=24 atrous convolution."""
    rates = [1, 6, 24]
    grid_ns = [7, 9, 9]   # grid size for each (rate=1 needs smaller grid)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
    fig.subplots_adjust(wspace=0.10, left=0.02, right=0.98, top=0.82, bottom=0.06)

    for ax, rate, gn in zip(axes, rates, grid_ns):
        _draw_atrous_panel(ax, rate, grid_n=gn)

    fig.suptitle(
        "Atrous (dilated) convolution — 3×3 kernel at different dilation rates.\n"
        "Standard convolution corresponds to rate = 1.",
        fontsize=8.5, y=0.97,
    )
    _save(fig, "atrous_convolution")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis architecture diagrams.")
    parser.add_argument("--only", choices=["unet", "deeplab", "atrous"],
                        help="Generate only one figure (default: all three)")
    args = parser.parse_args()

    which = args.only or "all"
    print(f"Writing to {OUT}/")

    if which in ("unet", "all"):
        print("Generating U-Net diagram …")
        plot_unet()

    if which in ("deeplab", "all"):
        print("Generating DeepLab variants diagram …")
        plot_deeplab_variants()

    if which in ("atrous", "all"):
        print("Generating atrous convolution diagram …")
        plot_atrous_convolution()

    print("Done.")


if __name__ == "__main__":
    main()
