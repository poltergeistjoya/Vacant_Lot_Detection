"""Collect metrics from all model runs into a single CSV.

Usage:
  uv run python scripts/collect_runs.py
  uv run python scripts/collect_runs.py --output runs.csv
  uv run python scripts/collect_runs.py --models-dir /zooper2/joya.debi/Vacant_Lot_Detection/outputs/models
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


COLUMNS = [
    "model_type", "encoder", "run_id", "best_epoch", "epochs_ran",
    "in_channels", "use_building_prob", "patch_size", "batch_size", "lr",
    "pos_weight", "bce_weight", "dice_weight", "lovasz_weight",
    "oversample_factor", "min_vacant_pixels", "band_dropout_p",
    "cosine_t_max", "seed", "eval_stride",
    "vacancy_mask", "patch_splits",
    "val_iou", "val_f1", "val_f2", "val_precision", "val_recall", "val_ap", "val_kappa",
    "test_iou", "test_f1", "test_f2", "test_precision", "test_recall", "test_ap", "test_kappa",
    "threshold_sweep_extent",
    "note", "trained_at", "machine",
]


def extract_run(run_dir: Path, machine: str) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    m = json.loads(metrics_path.read_text())
    met = m.get("metrics", {})
    val = met.get("val", {})
    test = met.get("test", {})
    training = m.get("training", {})
    loss = m.get("loss", {})
    model = m.get("model", {})

    # Try to get data paths from config.yaml
    config_path = run_dir / "config.yaml"
    vacancy_mask = ""
    patch_splits = ""
    patch_size = ""
    eval_stride = ""
    if config_path.exists():
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        dp = cfg.get("data_paths", {})
        vacancy_mask = dp.get("vacancy_mask", "")
        patch_splits = dp.get("patch_splits", "")
        patch_size = cfg.get("patch", {}).get("size", "")
        eval_stride = cfg.get("eval_stride", "")

    # Infer patch size from splits filename if not in config
    if not patch_size and patch_splits:
        for tok in Path(patch_splits).stem.split("_"):
            if tok.isdigit():
                patch_size = tok
                break

    # Threshold sweep extent from pr_curves
    sweep_extent = ""
    pr_path = run_dir / "pr_curves.npz"
    if pr_path.exists():
        try:
            import numpy as np
            data = np.load(pr_path)
            for key in ["val_thresholds", "thresholds_val"]:
                if key in data:
                    sweep_extent = f"{float(data[key].max()):.3f}"
                    break
        except Exception:
            pass

    # Epochs ran from history
    epochs_ran = ""
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        try:
            h = json.loads(hist_path.read_text())
            epochs_ran = len(h)
        except Exception:
            pass

    return {
        "model_type": m.get("model_type", model.get("type", "")),
        "encoder": m.get("encoder_name", model.get("encoder_name", "")),
        "run_id": m.get("run_id", run_dir.name),
        "best_epoch": m.get("best_epoch", ""),
        "epochs_ran": epochs_ran,
        "in_channels": model.get("in_channels", ""),
        "use_building_prob": model.get("use_building_prob", ""),
        "patch_size": patch_size,
        "batch_size": training.get("batch_size", ""),
        "lr": training.get("learning_rate", ""),
        "pos_weight": loss.get("pos_weight", ""),
        "bce_weight": loss.get("bce_weight", ""),
        "dice_weight": loss.get("dice_weight", ""),
        "lovasz_weight": loss.get("lovasz_weight", ""),
        "oversample_factor": training.get("oversample_factor", ""),
        "min_vacant_pixels": training.get("min_vacant_pixels", ""),
        "band_dropout_p": training.get("band_dropout_p", ""),
        "cosine_t_max": training.get("cosine_t_max", ""),
        "seed": training.get("seed", ""),
        "eval_stride": eval_stride,
        "vacancy_mask": vacancy_mask,
        "patch_splits": patch_splits,
        "val_iou": f"{val['iou']:.4f}" if "iou" in val else "",
        "val_f1": f"{val['f1']:.4f}" if "f1" in val else "",
        "val_f2": f"{val['f2']:.4f}" if "f2" in val else "",
        "val_precision": f"{val['precision']:.4f}" if "precision" in val else "",
        "val_recall": f"{val['recall']:.4f}" if "recall" in val else "",
        "val_ap": f"{val['average_precision']:.4f}" if "average_precision" in val else "",
        "val_kappa": f"{val['kappa']:.4f}" if "kappa" in val else "",
        "test_iou": f"{test['iou']:.4f}" if "iou" in test else "",
        "test_f1": f"{test['f1']:.4f}" if "f1" in test else "",
        "test_f2": f"{test['f2']:.4f}" if "f2" in test else "",
        "test_precision": f"{test['precision']:.4f}" if "precision" in test else "",
        "test_recall": f"{test['recall']:.4f}" if "recall" in test else "",
        "test_ap": f"{test['average_precision']:.4f}" if "average_precision" in test else "",
        "test_kappa": f"{test['kappa']:.4f}" if "kappa" in test else "",
        "threshold_sweep_extent": sweep_extent,
        "note": m.get("note", ""),
        "trained_at": m.get("trained_at", ""),
        "machine": machine,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect model run metrics into CSV")
    parser.add_argument("--models-dir", default=None,
                        help="Path to outputs/models/ (default: auto-detect from shared root)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: stdout)")
    parser.add_argument("--machine", default=None, help="Machine name tag (default: hostname)")
    args = parser.parse_args()

    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        # Auto-detect
        script_dir = Path(__file__).resolve().parent
        models_dir = script_dir.parent.parent / "outputs" / "models"
        if not models_dir.exists():
            models_dir = script_dir.parent / "outputs" / "models"

    if args.machine:
        machine = args.machine
    else:
        import socket
        machine = socket.gethostname()

    rows = []
    for arch_dir in sorted(models_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        for run_dir in sorted(arch_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            row = extract_run(run_dir, machine)
            if row:
                rows.append(row)

    if args.output:
        out = open(args.output, "w", newline="")
    else:
        out = sys.stdout

    writer = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    if args.output:
        out.close()
        print(f"Wrote {len(rows)} runs to {args.output}", file=sys.stderr)
    else:
        print(f"\n# {len(rows)} runs collected", file=sys.stderr)


if __name__ == "__main__":
    main()
