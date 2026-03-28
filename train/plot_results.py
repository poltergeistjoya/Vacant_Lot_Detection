"""
Generate figures for a trained pixel-level classifier.

Reads model.joblib + metrics.json from a flat run directory and writes PNGs to
the run's figures/ subdirectory:
  - pr_curve.png           — Precision-Recall curve for val and test
  - threshold_sweep.png    — F1/F2/Precision/Recall vs threshold for val
  - feature_importance.png — Top-10 feature importances
  - confusion_matrix.png   — Confusion matrices for val and test

Usage:
  uv run python train/plot_results.py --run outputs/models/rf/001
  uv run python train/plot_results.py --run outputs/models/gbm/001
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np

from vacant_lot.config import _get_shared_root
from vacant_lot.modeling import FEATURE_NAMES


def plot_pr_curve(metrics: dict, pr_curves: np.lib.npyio.NpzFile, figures_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"val": "#2196F3", "test": "#F44336"}
    for split in ("val", "test"):
        if split not in metrics["metrics"]:
            continue
        prec = pr_curves[f"{split}_pr_precision"]
        rec = pr_curves[f"{split}_pr_recall"]
        ap = metrics["metrics"][split]["average_precision"]
        ax.plot(rec, prec, label=f"{split} (AP={ap:.3f})", color=colors[split], linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {metrics['model_type']} run {metrics['run_id']}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    path = figures_dir / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_threshold_sweep(metrics: dict, pr_curves: np.lib.npyio.NpzFile, figures_dir) -> None:
    """Plot F1, F2, Precision, Recall vs threshold using val split PR curve data."""
    if "val_pr_precision" not in pr_curves:
        print("No val PR curve data, skipping threshold sweep.")
        return

    prec = pr_curves["val_pr_precision"][:-1]  # sklearn appends a trailing point
    rec = pr_curves["val_pr_recall"][:-1]
    thresh = pr_curves["val_pr_thresholds"]

    denom_f1 = np.maximum(prec + rec, 1e-8)
    denom_f2 = np.maximum(4 * prec + rec, 1e-8)
    f1 = 2 * prec * rec / denom_f1
    f2 = 5 * prec * rec / denom_f2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresh, prec, label="Precision", color="#2196F3", linewidth=1.5)
    ax.plot(thresh, rec,  label="Recall",    color="#F44336", linewidth=1.5)
    ax.plot(thresh, f1,   label="F1",        color="#4CAF50", linewidth=2)
    ax.plot(thresh, f2,   label="F2 (recall-weighted)", color="#FF9800", linewidth=2, linestyle="--")

    best_f1_idx = np.argmax(f1)
    best_f2_idx = np.argmax(f2)
    ax.axvline(thresh[best_f1_idx], color="#4CAF50", alpha=0.4, linestyle=":")
    ax.axvline(thresh[best_f2_idx], color="#FF9800", alpha=0.4, linestyle=":")
    ax.text(thresh[best_f1_idx], 0.02, f"F1={f1[best_f1_idx]:.3f}\n@{thresh[best_f1_idx]:.3f}",
            color="#4CAF50", fontsize=8, ha="center")
    ax.text(thresh[best_f2_idx], 0.12, f"F2={f2[best_f2_idx]:.3f}\n@{thresh[best_f2_idx]:.3f}",
            color="#FF9800", fontsize=8, ha="center")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sweep (val) — {metrics['model_type']} run {metrics['run_id']}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    path = figures_dir / "threshold_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_importance(model, metrics: dict, figures_dir) -> None:
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_, skipping.")
        return

    importances = model.feature_importances_
    # Normalize
    importances = importances / importances.sum()
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(FEATURE_NAMES)), importances[order], color="#4CAF50")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("Relative Importance")
    ax.set_title(f"Feature Importance — {metrics['model_type']} run {metrics['run_id']}")
    ax.grid(True, axis="y", alpha=0.3)

    path = figures_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrices(metrics: dict, figures_dir) -> None:
    splits = [s for s in ("val", "test") if s in metrics["metrics"]]
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        m = metrics["metrics"][split]
        cm = np.array(m["confusion_matrix"])  # [[TN, FP], [FN, TP]]
        total = cm.sum()

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{split}  (F1={m['f1']:.3f}, IoU={m['iou']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-vacant", "Vacant"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-vacant", "Vacant"])

        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.colorbar(im, ax=ax)

    fig.suptitle(f"{metrics['model_type']} run {metrics['run_id']}", fontsize=12)
    plt.tight_layout()

    path = figures_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results for a trained model run")
    parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory (e.g. outputs/models/rf/001)",
    )
    args = parser.parse_args()

    shared_root = _get_shared_root()
    run_dir = shared_root / args.run
    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    pr_curves_path = run_dir / "pr_curves.npz"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    if not pr_curves_path.exists():
        raise FileNotFoundError(f"PR curves not found: {pr_curves_path}")

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text())
    pr_curves = np.load(pr_curves_path)

    print(f"Model type: {metrics['model_type']}, run: {metrics['run_id']}")
    for split, m in metrics["metrics"].items():
        print(f"  {split}: IoU={m['iou']:.4f}, F1={m['f1']:.4f}, AP={m['average_precision']:.4f}")

    plot_pr_curve(metrics, pr_curves, figures_dir)
    plot_threshold_sweep(metrics, pr_curves, figures_dir)
    plot_feature_importance(model, metrics, figures_dir)
    plot_confusion_matrices(metrics, figures_dir)


if __name__ == "__main__":
    main()
