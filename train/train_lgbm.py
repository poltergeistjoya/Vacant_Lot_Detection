"""
Train a pixel-level LightGBM baseline on NAIP imagery.

Samples pixels from train patches with reservoir sampling, corrects class
weights for the true vacant/non-vacant distribution, trains a LGBM classifier,
then streams evaluation on val and test splits.

Usage:
  uv run python scripts/train_lgbm.py
  uv run python scripts/train_lgbm.py --config config/lgbm.yaml
  uv run python scripts/train_lgbm.py --run-id 003
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

from vacant_lot.config import LGBMModelConfig, load_train_config
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
    parser = argparse.ArgumentParser(description="Train pixel-level LightGBM baseline")
    parser.add_argument(
        "--config",
        default="lgbm.yaml",
        help="Path to LightGBM training config YAML (default: config/lgbm.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (default: auto-increment from output_dir)",
    )
    args = parser.parse_args()

    data_cfg, train_cfg = load_train_config(args.config)

    if not isinstance(train_cfg.model, LGBMModelConfig):
        log.error(f"Expected LightGBM config, got {type(train_cfg.model)}")
        sys.exit(1)

    model_cfg = train_cfg.model
    sampling_cfg = train_cfg.sampling
    shared_root = data_cfg._shared_root
    note = os.environ.get("VACANT_LOT_RUN_NOTE", train_cfg.note)

    base_output_dir = shared_root / train_cfg.output_dir
    run_id = args.run_id or next_run_id(base_output_dir)
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Run:           gbm/{run_id}")
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

    splits = load_patch_splits(splits_path)
    patch_size = data_cfg.patch.size

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
    # Compute scale_pos_weight to correct for undersampling
    # -------------------------------------------------------------------
    # LightGBM uses scale_pos_weight = (neg_weight / pos_weight) as a single
    # scalar, equivalent to the ratio of class weights.
    # pos_weight  = true_vacant / sampled_vacant
    # neg_weight  = true_nonvacant / sampled_nonvacant
    # scale_pos_weight = neg_weight / pos_weight
    scale_pos_weight = sampling_cfg.nonvacant_weight / sampling_cfg.vacant_weight
    log.info(
        f"scale_pos_weight={scale_pos_weight:.2f} "
        f"(true distribution: {sampling_cfg.true_vacant_count / (sampling_cfg.true_vacant_count + sampling_cfg.true_nonvacant_count) * 100:.1f}% vacant)"
    )

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed. Run: uv add lightgbm")
        sys.exit(1)

    log.info(
        f"Training LightGBM: n_estimators={model_cfg.n_estimators}, "
        f"max_depth={model_cfg.max_depth}, "
        f"learning_rate={model_cfg.learning_rate}, "
        f"num_leaves={model_cfg.num_leaves}"
    )
    clf = lgb.LGBMClassifier(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        num_leaves=model_cfg.num_leaves,
        min_child_samples=model_cfg.min_child_samples,
        n_jobs=model_cfg.n_jobs,
        random_state=model_cfg.random_state,
        scale_pos_weight=scale_pos_weight,
    )
    from tqdm import tqdm

    with tqdm(total=model_cfg.n_estimators, desc="Training", unit="round") as pbar:
        def _progress(env):
            pbar.update(1)

        clf.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0), _progress])
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
            model_name=f"gbm/{run_id}/{split_name}",
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
        "model_type": "lightgbm",
        "trained_at": datetime.now().isoformat(),
        "config_file": args.config,
        "note": note,
        "model": model_cfg.model_dump(),
        "sampling": sampling_cfg.model_dump(),
        "scale_pos_weight": scale_pos_weight,
        "features": FEATURE_NAMES,
        "n_train_patches": len(splits["train"]),
        "metrics": all_metrics,
    }

    save_model(clf, run_dir / "model.joblib", metadata=metadata)
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))
    np.savez(run_dir / "pr_curves.npz", **pr_curves)
    log.info(f"PR curves written to {run_dir / 'pr_curves.npz'}")
    log.info(f"Metrics written to {run_dir / 'metrics.json'}")

    log.info("--- Summary ---")
    for split_name, m in all_metrics.items():
        log.info(
            f"  {split_name}: IoU={m['iou']:.4f}, F1={m['f1']:.4f}, F2={m['f2']:.4f}, "
            f"AP={m['average_precision']:.4f}, Prec={m['precision']:.4f}, "
            f"Rec={m['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
