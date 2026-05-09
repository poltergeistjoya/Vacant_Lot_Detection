"""
Compute probability calibration for a trained DL segmentation model.

Fits temperature scaling and Platt scaling on val (Bronx) logits, evaluates
Expected Calibration Error before and after on val and test (Brooklyn), and
saves reliability diagrams.

Usage:
  uv run python scripts/calibrate_model.py --run-dir outputs/models/deeplabv3plus/kahan_027
  uv run python scripts/calibrate_model.py --run-dir outputs/models/deeplabv3plus/kahan_027 --n-samples 200000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vacant_lot.config import _get_shared_root, load_data_config
from vacant_lot.dataset import NAIPSegmentationDataset, load_patch_splits
from vacant_lot.segmentation import build_model
from vacant_lot.train import _auto_device
from vacant_lot.logger import get_logger

log = get_logger()

mpl.rcParams.update({
    "font.family":      "STIX Two Text",
    "mathtext.fontset": "stix",
    "font.size":        8,
})


def _style_ax(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------

def collect_logits(
    model: torch.nn.Module,
    dataset: NAIPSegmentationDataset,
    device: torch.device,
    n_pos: int = 75_000,
    n_neg: int = 75_000,
    split_name: str = "",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Collect raw logits + labels from a dataset using stratified reservoir sampling.

    Returns:
        logits: float32 array (n,) — raw model output before sigmoid
        labels: int8 array (n,)
        total_pos_seen: total positive pixels encountered across all patches
        total_neg_seen: total negative pixels encountered across all patches
    """
    model.eval()
    rng = np.random.default_rng(42)

    res_pos = np.empty(n_pos, dtype=np.float32)
    res_neg = np.empty(n_neg, dtype=np.float32)
    pos_count = neg_count = 0
    total_pos = total_neg = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def _reservoir_update(
        batch: np.ndarray, buf: np.ndarray, count: int, total: int, cap: int
    ) -> tuple[int, int]:
        n = len(batch)
        if n == 0:
            return count, total
        if count < cap:
            space = min(n, cap - count)
            buf[count : count + space] = batch[:space]
            count += space
            total += space
            batch = batch[space:]
            n = len(batch)
        if n > 0:
            # For each new item, draw a random slot; accept if slot < cap.
            slots = rng.integers(0, np.arange(total, total + n) + 1)
            accept = slots < cap
            if accept.any():
                buf[slots[accept]] = batch[accept]
            total += n
        return count, total

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Collect logits {split_name}", unit="patch"):
            images = images.to(device)
            # model output: (1, 1, H, W) → squeeze → (H, W) raw logits
            raw = model(images).squeeze().cpu().numpy().astype(np.float32)
            mask_2d = masks.squeeze().numpy()

            valid = mask_2d != 255
            if not valid.any():
                continue

            lv = raw[valid]
            lab = mask_2d[valid] == 1

            pos_count, total_pos = _reservoir_update(lv[lab], res_pos, pos_count, total_pos, n_pos)
            neg_count, total_neg = _reservoir_update(lv[~lab], res_neg, neg_count, total_neg, n_neg)

    logits = np.concatenate([res_pos[:pos_count], res_neg[:neg_count]])
    labels = np.concatenate([
        np.ones(pos_count, dtype=np.int8),
        np.zeros(neg_count, dtype=np.int8),
    ])

    log.info(
        f"{split_name}: {pos_count:,} pos / {neg_count:,} neg in reservoir "
        f"(seen: {total_pos:,} pos / {total_neg:,} neg)"
    )
    return logits, labels, total_pos, total_neg


# ---------------------------------------------------------------------------
# Calibration fitting
# ---------------------------------------------------------------------------

def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray
) -> float:
    """Find scalar temperature T that minimises weighted NLL of sigmoid(logit / T).

    sample_weights must reflect the TRUE class distribution (not the balanced
    reservoir).  Fitting without weights assumes a 50/50 prior and finds a T
    that is far too large for a 3.4% vacancy rate.
    """
    w = sample_weights / sample_weights.sum()

    def weighted_nll(T: float) -> float:
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        per_sample = -(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        return float(np.dot(w, per_sample))

    result = minimize_scalar(weighted_nll, bounds=(0.05, 20.0), method="bounded")
    T = float(result.x)
    log.info(f"Temperature: T={T:.4f}  wNLL {weighted_nll(1.0):.4f} → {weighted_nll(T):.4f}")
    return T


def fit_platt(
    logits: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray
) -> LogisticRegression:
    """Fit Platt scaling using sample-weighted logistic regression.

    sample_weights must reflect the TRUE class distribution so the intercept
    is not biased by the balanced reservoir.
    """
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logits.reshape(-1, 1), labels, sample_weight=sample_weights)
    log.info(f"Platt: a={lr.coef_[0][0]:.4f}, b={lr.intercept_[0]:.4f}")
    return lr


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    n_bins: int = 15,
    min_samples: int = 50,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted ECE and per-bin stats.

    Weights correct for the stratified reservoir sampling so statistics
    reflect the true class distribution.

    Returns:
        ece: scalar ECE
        bin_mean_prob: (n_bins,) actual mean predicted prob per bin (nan if empty/sparse)
        bin_accs: (n_bins,) weighted positive rate per bin (nan if empty/sparse)
        bin_fracs: (n_bins,) fraction of total weight per bin
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)

    w_total = sample_weights.sum()
    bin_mean_prob = np.full(n_bins, np.nan)
    bin_accs = np.full(n_bins, np.nan)
    bin_fracs = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() < min_samples:
            continue
        w = sample_weights[mask]
        w_sum = w.sum()
        bin_mean_prob[b] = float(np.average(probs[mask], weights=w))
        bin_accs[b] = float((w * labels[mask]).sum() / w_sum)
        bin_fracs[b] = float(w_sum / w_total)

    valid = ~np.isnan(bin_accs)
    ece = float((bin_fracs[valid] * np.abs(bin_mean_prob[valid] - bin_accs[valid])).sum())
    return ece, bin_mean_prob, bin_accs, bin_fracs


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability(
    logits: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    temperature: float,
    platt: LogisticRegression,
    n_bins: int = 15,
    title: str = "",
    out_path: Path | None = None,
) -> dict[str, float]:
    """Plot reliability diagram comparing raw, temperature, and Platt calibration."""
    probs_raw   = sigmoid(logits)
    probs_temp  = sigmoid(logits / temperature)
    probs_platt = platt.predict_proba(logits.reshape(-1, 1))[:, 1]

    ece_raw,   bc_raw,   acc_raw,   frac_raw   = compute_ece(probs_raw,   labels, sample_weights, n_bins)
    ece_temp,  bc_temp,  acc_temp,  frac_temp  = compute_ece(probs_temp,  labels, sample_weights, n_bins)
    ece_platt, bc_platt, acc_platt, frac_platt = compute_ece(probs_platt, labels, sample_weights, n_bins)

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax_main.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=0.8, label="Perfect")
    ax_main.plot(bc_raw,   acc_raw,   "o-", ms=3, lw=1.0, color="#e05c5c",
                 label=f"Raw (ECE={ece_raw:.3f})")
    ax_main.plot(bc_temp,  acc_temp,  "s-", ms=3, lw=1.0, color="#4477cc",
                 label=f"Temp T={temperature:.2f} (ECE={ece_temp:.3f})")
    ax_main.plot(bc_platt, acc_platt, "^-", ms=3, lw=1.0, color="#44aa77",
                 label=f"Platt (ECE={ece_platt:.3f})")

    ax_main.set_ylabel("Fraction positive", fontsize=8)
    ax_main.set_xlim(0.0, 1.0)
    ax_main.set_ylim(0.0, 1.0)
    ax_main.legend(fontsize=6, frameon=False, loc="upper left")
    if title:
        ax_main.set_title(title, fontsize=8)
    _style_ax(ax_main)

    bar_w = 0.8 / n_bins
    valid_raw = ~np.isnan(bc_raw)
    ax_hist.bar(bc_raw[valid_raw], frac_raw[valid_raw], width=bar_w, color="#e05c5c", alpha=0.6)
    ax_hist.set_xlabel("Mean predicted probability", fontsize=8)
    ax_hist.set_ylabel("Weight\nfraction", fontsize=7)
    ax_hist.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    _style_ax(ax_hist)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved {out_path}")
    plt.close(fig)

    return {"raw": ece_raw, "temperature": ece_temp, "platt": ece_platt}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_weights(
    labels: np.ndarray, total_pos: int, total_neg: int
) -> np.ndarray:
    """Inverse-frequency weights that de-bias the stratified reservoir back to true prior."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    w = np.empty(len(labels), dtype=np.float64)
    w[labels == 1] = total_pos / max(n_pos, 1)
    w[labels == 0] = total_neg / max(n_neg, 1)
    return w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a trained DL segmentation model")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to run directory, relative to shared root "
             "(e.g. outputs/models/deeplabv3plus/kahan_027)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=150_000,
        help="Total reservoir size, split equally between pos and neg (default: 150000)"
    )
    parser.add_argument(
        "--n-bins", type=int, default=15,
        help="Number of equal-width probability bins for reliability diagram (default: 15)"
    )
    args = parser.parse_args()

    shared_root = _get_shared_root()
    run_dir = shared_root / args.run_dir
    if not run_dir.exists():
        log.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Load config saved alongside the checkpoint
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg      = cfg["model"]
    arch           = model_cfg["type"]
    encoder_name   = model_cfg["encoder_name"]
    encoder_weights = model_cfg.get("encoder_weights")
    in_channels    = model_cfg["in_channels"]
    use_building   = model_cfg.get("use_building_prob", False)

    data_paths        = cfg["data_paths"]
    vrt_path          = shared_root / data_paths["vrt"]
    vacancy_mask_path = shared_root / data_paths["vacancy_mask"]
    splits_path       = shared_root / data_paths["patch_splits"]

    # Always load data.yaml — needed for split→borough labels and optionally building_pred path
    _BORO_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
    data_cfg = load_data_config()
    val_label  = " + ".join(_BORO_NAMES.get(b, str(b)) for b in data_cfg.split.val_boroughs)
    test_label = " + ".join(_BORO_NAMES.get(b, str(b)) for b in data_cfg.split.test_boroughs)
    log.info(f"Split labels: val={val_label}, test={test_label}")

    building_pred_path = None
    if use_building:
        building_pred_path = data_cfg.get_building_pred_path()
        if not building_pred_path.exists():
            log.error(f"building_pred.tif not found: {building_pred_path}")
            sys.exit(1)
        log.info(f"Building prob channel: {building_pred_path}")

    decoder_channels = model_cfg.get("decoder_channels")  # UNet only; None → smp default

    # Build model — pass encoder_weights=None since we're loading our own weights
    device = _auto_device()
    model = build_model(
        arch=arch,
        in_channels=in_channels,
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=1,
        decoder_channels=decoder_channels,
    )
    best_pt = run_dir / "best.pt"
    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    log.info(f"Loaded {best_pt} (epoch {ckpt.get('epoch', '?')})")

    splits, splits_meta = load_patch_splits(splits_path)
    patch_size = splits_meta["patch_size"]

    n_half = args.n_samples // 2

    def _make_dataset(split_name: str) -> NAIPSegmentationDataset:
        return NAIPSegmentationDataset(
            vrt_path=vrt_path,
            vacancy_mask_path=vacancy_mask_path,
            patch_coords=splits[split_name],
            patch_size=patch_size,
            in_channels=in_channels,
            building_pred_path=building_pred_path,
            use_building_prob=use_building,
        )

    # Collect logits — val is used to fit, test is held out for evaluation
    val_logits, val_labels, val_total_pos, val_total_neg = collect_logits(
        model, _make_dataset("val"), device,
        n_pos=n_half, n_neg=n_half, split_name="val",
    )
    test_logits, test_labels, test_total_pos, test_total_neg = collect_logits(
        model, _make_dataset("test"), device,
        n_pos=n_half, n_neg=n_half, split_name="test",
    )

    val_weights  = _sample_weights(val_labels,  val_total_pos,  val_total_neg)
    test_weights = _sample_weights(test_labels, test_total_pos, test_total_neg)

    # Fit calibrators on val, weighted to reflect true class distribution
    temperature = fit_temperature(val_logits, val_labels, val_weights)
    platt = fit_platt(val_logits, val_labels, val_weights)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    val_ece = plot_reliability(
        val_logits, val_labels, val_weights,
        temperature, platt, n_bins=args.n_bins,
        title=f"Reliability — val ({val_label})   {arch}/{run_dir.name}",
        out_path=fig_dir / "calibration_val.png",
    )
    test_ece = plot_reliability(
        test_logits, test_labels, test_weights,
        temperature, platt, n_bins=args.n_bins,
        title=f"Reliability — test ({test_label})   {arch}/{run_dir.name}",
        out_path=fig_dir / "calibration_test.png",
    )

    calibration_out = {
        "run_dir": str(run_dir),
        "temperature": temperature,
        "platt_coef": float(platt.coef_[0][0]),
        "platt_intercept": float(platt.intercept_[0]),
        "ece": {"val": val_ece, "test": test_ece},
        "n_samples": args.n_samples,
        "n_bins": args.n_bins,
    }
    out_json = run_dir / "calibration.json"
    out_json.write_text(json.dumps(calibration_out, indent=2))
    log.info(f"Saved {out_json}")

    log.info("--- Summary ---")
    log.info(f"  Temperature T = {temperature:.4f}")
    log.info(f"  Platt: a={platt.coef_[0][0]:.4f}, b={platt.intercept_[0]:.4f}")
    for sn, ece_d in [("val", val_ece), ("test", test_ece)]:
        log.info(
            f"  {sn}: ECE raw={ece_d['raw']:.4f}, "
            f"temp={ece_d['temperature']:.4f}, "
            f"platt={ece_d['platt']:.4f}"
        )


if __name__ == "__main__":
    main()
