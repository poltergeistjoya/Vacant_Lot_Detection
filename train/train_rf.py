"""
Train a pixel-level Random Forest baseline on NAIP imagery.

Samples pixels from train patches with reservoir sampling, corrects class
weights for the true vacant/non-vacant distribution, trains an RF, then
streams evaluation on val and test splits.

Usage:
  uv run python scripts/train_rf.py
  uv run python scripts/train_rf.py --config config/rf.yaml
  uv run python scripts/train_rf.py --run-id 003
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from vacant_lot.config import RFModelConfig, load_train_config
from vacant_lot.dataset import load_patch_splits
from vacant_lot.logger import get_logger
from vacant_lot.modeling import (
    FEATURE_NAMES,
    evaluate_segmentation_streaming,
    sample_pixels_from_patches,
    save_model,
)

log = get_logger()


def next_run_id(output_dir: Path) -> str:
    """Return the next zero-padded 3-digit run ID by scanning output_dir."""
    if not output_dir.exists():
        return "001"
    existing = sorted(
        int(p.name) for p in output_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    return f"{(existing[-1] + 1 if existing else 1):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pixel-level RF baseline")
    parser.add_argument(
        "--config",
        default="rf.yaml",
        help="Path to RF training config YAML (default: config/rf.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (default: auto-increment from output_dir)",
    )
    args = parser.parse_args()

    data_cfg, train_cfg = load_train_config(args.config)

    if not isinstance(train_cfg.model, RFModelConfig):
        log.error(f"Expected RF config, got {type(train_cfg.model)}")
        sys.exit(1)

    model_cfg = train_cfg.model
    sampling_cfg = train_cfg.sampling
    shared_root = data_cfg._shared_root
    note = os.environ.get("VACANT_LOT_RUN_NOTE", train_cfg.note)

    base_output_dir = shared_root / train_cfg.output_dir
    run_id = args.run_id or next_run_id(base_output_dir)
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Run:           rf/{run_id}")
    log.info(f"Output dir:    {run_dir}")
    log.info(f"Config:        {args.config}")
    if note:
        log.info(f"Note:          {note}")

    vrt_path = data_cfg.get_vrt_path()
    vacancy_mask_path = data_cfg.get_vacancy_mask_path()
    splits_path = data_cfg.get_patch_splits_path()

    log.info(f"VRT:           {vrt_path}")
    log.info(f"Vacancy mask:  {vacancy_mask_path}")
    log.info(f"Splits:        {splits_path}")

    splits, patch_size = load_patch_splits(splits_path)

    # -------------------------------------------------------------------
    # Sample pixels from train patches
    # -------------------------------------------------------------------
    log.info(
        f"Sampling up to {sampling_cfg.n_vacant:,} vacant and "
        f"{sampling_cfg.n_nonvacant:,} non-vacant pixels from "
        f"{len(splits['train']):,} train patches..."
    )
    X_train, y_train = sample_pixels_from_patches(
        vrt_path=vrt_path,
        mask_path=vacancy_mask_path,
        patch_coords=splits["train"],
        n_vacant=sampling_cfg.n_vacant,
        n_nonvacant=sampling_cfg.n_nonvacant,
        patch_size=patch_size,
        random_state=sampling_cfg.random_state,
    )
    log.info(f"Train set: {len(y_train):,} pixels ({y_train.sum():,} vacant)")

    # -------------------------------------------------------------------
    # Compute class weights to correct for undersampling
    # -------------------------------------------------------------------
    class_weight = {
        1: sampling_cfg.vacant_weight,
        0: sampling_cfg.nonvacant_weight,
    }
    log.info(
        f"Class weights — vacant: {class_weight[1]:.2f}, "
        f"non-vacant: {class_weight[0]:.2f} "
        f"(true distribution: {sampling_cfg.true_vacant_count / (sampling_cfg.true_vacant_count + sampling_cfg.true_nonvacant_count) * 100:.1f}% vacant)"
    )

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    log.info(
        f"Training RF: n_estimators={model_cfg.n_estimators}, "
        f"max_depth={model_cfg.max_depth}, "
        f"min_samples_leaf={model_cfg.min_samples_leaf}, "
        f"n_jobs={model_cfg.n_jobs}"
    )
    clf = RandomForestClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        min_samples_leaf=model_cfg.min_samples_leaf,
        n_jobs=model_cfg.n_jobs,
        random_state=model_cfg.random_state,
        class_weight=class_weight,
    )
    clf.fit(X_train, y_train)
    log.info("Training complete.")

    del X_train, y_train
    gc.collect()

    # -------------------------------------------------------------------
    # Stream eval on val and test
    # -------------------------------------------------------------------
    _PR_KEYS = ("pr_precision", "pr_recall", "pr_thresholds")
    all_metrics: dict[str, dict] = {}
    pr_curves: dict[str, np.ndarray] = {}
    for split_name in ("val", "test"):
        log.info(f"Evaluating on {split_name} ({len(splits[split_name]):,} patches)...")
        m = evaluate_segmentation_streaming(
            model=clf,
            vrt_path=vrt_path,
            mask_path=vacancy_mask_path,
            patch_coords=splits[split_name],
            patch_size=patch_size,
            model_name=f"rf/{run_id}/{split_name}",
        )
        all_metrics[split_name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()
            if k not in _PR_KEYS
        }
        for k in _PR_KEYS:
            if k in m:
                pr_curves[f"{split_name}_{k}"] = m[k]

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    metadata = {
        "run_id": run_id,
        "model_type": "random_forest",
        "trained_at": datetime.now().isoformat(),
        "config_file": args.config,
        "note": note,
        "model": model_cfg.model_dump(),
        "sampling": sampling_cfg.model_dump(),
        "features": FEATURE_NAMES,
        "n_train_patches": len(splits["train"]),
        "metrics": all_metrics,
    }

    save_model(clf, run_dir / "model.joblib", metadata=metadata)
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))
    log.info(f"Metrics written to {run_dir / 'metrics.json'}")
    np.savez(run_dir / "pr_curves.npz", **pr_curves)
    log.info(f"PR curves written to {run_dir / 'pr_curves.npz'}")

    log.info("--- Summary ---")
    for split_name, m in all_metrics.items():
        log.info(
            f"  {split_name}: IoU={m['iou']:.4f}, F1={m['f1']:.4f}, F2={m['f2']:.4f}, "
            f"AP={m['average_precision']:.4f}, Prec={m['precision']:.4f}, "
            f"Rec={m['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
