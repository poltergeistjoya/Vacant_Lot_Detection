"""
Plot building permits per capita from 2000 onward for each MSA.

Produces two figures:
  1. Total permitted units per capita over time (line chart)
  2. Single- vs. multi-family composition over time (stacked-area, 2×2 panel)

Usage:
    uv run python scripts/plot_building_permits.py
    uv run python scripts/plot_building_permits.py --metric total_bldgs_per_capita
    uv run python scripts/plot_building_permits.py --out outputs/figures/permits.png
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

mpl.rcParams.update({
    "font.family":        "STIX Two Text",
    "mathtext.fontset":   "stix",
    "font.size":          8,
    "axes.titleweight":   "normal",
    "axes.labelweight":   "normal",
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SHARED_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = SHARED_ROOT / "data" / "housing" / "housing_data_comparisons.json"
DEFAULT_OUT = SHARED_ROOT / "outputs" / "figures" / "building_permits.png"

# ---------------------------------------------------------------------------
# MSA display names + explicit color assignment
# New York was orange by default → reassigned to purple
# ---------------------------------------------------------------------------
MSA_SHORT = {
    "New York-Newark-Jersey City MSA, NY-NJ":          "New York",
    "Philadelphia-Camden-Wilmington MSA, PA-NJ-DE-MD": "Philadelphia",
    "Dallas-Fort Worth-Arlington MSA, TX":             "Dallas–Fort Worth",
    "Phoenix-Mesa-Chandler MSA, AZ":                   "Phoenix",
}

MSA_COLOR = {
    "Dallas–Fort Worth": "#2166ac",   # blue
    "New York":          "#762a83",   # purple  (was default orange)
    "Philadelphia":      "#1b7837",   # green
    "Phoenix":           "#d6604d",   # muted red
}

METRIC_LABELS = {
    "total_units_per_capita":        "Total Permitted Units per Capita",
    "total_bldgs_per_capita":        "Total Permitted Buildings per Capita",
    "1_unit_units_per_capita":       "Single-Family Units per Capita",
    "5_plus_units_units_per_capita": "5+ Unit Buildings: Units per Capita",
}

# Stacked-area layers for the mix chart (bottom → top)
# Color gradient: red (single-family) → blue (5+ units)
MIX_LAYERS = [
    ("1_unit_units_per_capita",       "Single-family (1 unit)",  "#d6604d"),
    ("2_units_units_per_capita",      "2-unit",                  "#fddbc7"),
    ("3_to_4_units_units_per_capita", "3–4 unit",                "#92c5de"),
    ("5_plus_units_units_per_capita", "5+ unit",                 "#4393c3"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_ieee_style(ax):
    """Minimal, publication-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(labelsize=7, length=2, width=0.4)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#cccccc")
    ax.set_axisbelow(True)


def load_series(metric: str, start_year: int = 2000) -> dict[str, tuple[list, list]]:
    with open(DATA_PATH) as f:
        records = json.load(f)
    series: dict[str, tuple[list, list]] = {}
    for r in records:
        year = int(r["year"])
        if year < start_year:
            continue
        name = MSA_SHORT.get(r["name"], r["name"])
        val = r.get(metric)
        if val is None:
            continue
        if name not in series:
            series[name] = ([], [])
        series[name][0].append(year)
        series[name][1].append(val)
    return {
        k: (sorted(xs), [y for _, y in sorted(zip(xs, ys))])
        for k, (xs, ys) in series.items()
    }


def load_multi_series(start_year: int = 2000) -> dict[str, dict[str, tuple[list, list]]]:
    """Return {msa_name: {layer_key: (years, vals)}} for all mix layers."""
    with open(DATA_PATH) as f:
        records = json.load(f)
    out: dict[str, dict] = {}
    for r in records:
        year = int(r["year"])
        if year < start_year:
            continue
        name = MSA_SHORT.get(r["name"], r["name"])
        if name not in out:
            out[name] = {key: ([], []) for key, _, _ in MIX_LAYERS}
        for key, _, _ in MIX_LAYERS:
            val = r.get(key, 0) or 0
            out[name][key][0].append(year)
            out[name][key][1].append(val)
    # sort by year
    for name in out:
        for key in out[name]:
            xs, ys = out[name][key]
            paired = sorted(zip(xs, ys))
            out[name][key] = ([p[0] for p in paired], [p[1] for p in paired])
    return out


# ---------------------------------------------------------------------------
# Combined figure: (a) line chart left, (b) 2×2 mix right
# ---------------------------------------------------------------------------
def plot_combined(metric: str, out: Path, start_year: int = 2000):
    series = load_series(metric, start_year)
    mix_data = load_multi_series(start_year)
    ylabel = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    msas = ["Dallas–Fort Worth", "Phoenix", "New York", "Philadelphia"]

    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 2], wspace=0.15)

    # --- (a) line chart ---
    ax_left = fig.add_subplot(gs[0])
    for name in sorted(series):
        years, vals = series[name]
        ax_left.plot(
            years, vals,
            marker="o", markersize=2, linewidth=0.9,
            markeredgewidth=0, label=name, color=MSA_COLOR.get(name, None),
        )
    ax_left.axvline(2008, color="#999999", linewidth=0.5, linestyle="--", zorder=0)
    ax_left.text(2008.3, ax_left.get_ylim()[1], "Great Recession",
                 fontsize=6, color="#999999", va="top")
    ax_left.set_xlabel("Year", fontsize=8)
    ax_left.set_ylabel(ylabel, fontsize=8)
    ax_left.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.4f}"))
    ax_left.set_xlim(left=start_year)
    ax_left.legend(fontsize=7, frameon=False)
    _apply_ieee_style(ax_left)

    # --- (b) 2×2 mix panels ---
    gs_right = gs[1].subgridspec(2, 2)
    legend_handles = None
    right_axes = []

    for i, name in enumerate(msas):
        ax = fig.add_subplot(gs_right[i // 2, i % 2])
        right_axes.append(ax)
        layers = mix_data[name]
        years = np.array(layers[MIX_LAYERS[0][0]][0])
        stack = np.array([layers[key][1] for key, _, _ in MIX_LAYERS])
        colors = [c for _, _, c in MIX_LAYERS]
        labels = [lbl for _, lbl, _ in MIX_LAYERS]

        polys = ax.stackplot(years, stack, labels=labels, colors=colors, alpha=0.7, linewidth=0)
        ax.axvline(2008, color="#999999", linewidth=0.5, linestyle="--", zorder=0)
        ax.set_title(name, fontsize=8)
        ax.set_xlim(left=start_year)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.4f}"))
        # y-label only on left column
        if i % 2 == 0:
            ax.set_ylabel("Permitted Units per Capita", fontsize=7)
        ax.set_xlabel("Year", fontsize=7)
        _apply_ieee_style(ax)
        ax.tick_params(axis="x", labelsize=7, labelbottom=True)

        if legend_handles is None:
            legend_handles = polys
            legend_labels = labels

    # Shared y baseline for captions — bottom of the lowest panel on either side
    y_bot = min(ax_left.get_position().y0,
                min(ax.get_position().y0 for ax in right_axes))
    y_caption = y_bot - 0.13

    # (a) centered under left panel
    x_a = (ax_left.get_position().x0 + ax_left.get_position().x1) / 2
    fig.text(x_a, y_caption, r"$\mathbf{(a)}$", ha="center", va="top", fontsize=9)

    # (b) centered under the 2×2 block
    x_left  = right_axes[2].get_position().x0
    x_right = right_axes[3].get_position().x1
    x_center_b = (right_axes[2].get_position().x0 +
              right_axes[3].get_position().x1) / 2
    fig.text((x_left + x_right) / 2, y_caption, r"$\mathbf{(b)}$",
             ha="center", va="top", fontsize=9)

    fig.legend(legend_handles, legend_labels,
               loc="upper center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(x_center_b,
                               min(ax.get_position().y0 for ax in right_axes) - 0.09))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="total_units_per_capita",
                        choices=list(METRIC_LABELS),
                        help="Per-capita metric for the line chart (panel a)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output PNG path")
    parser.add_argument("--start-year", type=int, default=2000)
    args = parser.parse_args()

    plot_combined(args.metric, args.out, args.start_year)


if __name__ == "__main__":
    main()
