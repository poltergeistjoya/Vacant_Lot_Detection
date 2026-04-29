"""
Post-training evaluation for a completed (or killed) run.

Loads best.pt from a run directory, evaluates on val and test splits,
and writes metrics.json + pr_curves.npz — identical output to what
train_dl.py produces at the end of a successful training run.

Use this when a run was killed early (before post-training eval ran),
or when --resume reset the patience counter and the run ran too long.

Usage:
  uv run python train/eval_run.py --run outputs/models/unet/kahan_038
  uv run python train/eval_run.py --run outputs/models/unet/kahan_038 --eval-stride 512
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from vacant_lot.config import _get_shared_root, load_data_config
from vacant_lot.dataset import (
    NAIPSegmentationDataset,
    generate_overlap_splits,
    load_patch_splits,
)
from vacant_lot.logger import get_logger
from vacant_lot.segmentation import build_model
from vacant_lot.train import _auto_device

# Re-use eval functions from train_dl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dl import evaluate_dl_overlap, evaluate_dl_streaming

log = get_logger()

_PR_KEYS = ("pr_precision", "pr_recall", "pr_thresholds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best.pt for a completed run")
    parser.add_argument("--run", required=True,
                        help="Run directory relative to shared root "
                             "(e.g. outputs/models/unet/kahan_038)")
    parser.add_argument("--eval-stride", type=int, default=None,
                        help="Inference stride for overlap eval. Reads from config if not set.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Splits to evaluate (default: val test)")
    args = parser.parse_args()

    shared_root = _get_shared_root()
    run_dir = shared_root / args.run
    if not run_dir.exists():
        log.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        log.error(f"best.pt not found: {best_pt}")
        sys.exit(1)

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_cfg_raw  = cfg["model"]
    training_raw   = cfg["training"]
    loss_raw       = cfg["loss"]
    data_paths_raw = cfg["data_paths"]
    note           = cfg.get("note", "")

    arch             = model_cfg_raw["type"]
    encoder_name     = model_cfg_raw["encoder_name"]
    encoder_weights  = model_cfg_raw.get("encoder_weights")
    in_channels      = model_cfg_raw["in_channels"]
    use_building     = model_cfg_raw.get("use_building_prob", False)
    decoder_channels = model_cfg_raw.get("decoder_channels")
    classes          = model_cfg_raw.get("classes", 1)

    cfg_eval_stride = cfg.get("eval_stride")
    eval_stride = args.eval_stride if args.eval_stride is not None else cfg_eval_stride

    vrt_path          = shared_root / data_paths_raw["vrt"]
    vacancy_mask_path = shared_root / data_paths_raw["vacancy_mask"]
    borough_mask_path = shared_root / data_paths_raw["borough_mask"]
    splits_path       = shared_root / data_paths_raw["patch_splits"]

    building_pred_path = None
    if use_building:
        data_cfg = load_data_config()
        building_pred_path = data_cfg.get_building_pred_path()
        if not building_pred_path.exists():
            log.error(f"building_pred.tif not found: {building_pred_path}")
            sys.exit(1)
        log.info(f"Building prob: {building_pred_path}")

    splits, splits_meta = load_patch_splits(splits_path)
    patch_size = splits_meta["patch_size"]
    run_id = run_dir.name

    use_overlap = eval_stride is not None and eval_stride < patch_size

    device = _auto_device()
    log.info(f"Device: {device}")

    model = build_model(
        arch=arch,
        in_channels=in_channels,
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=classes,
        decoder_channels=decoder_channels,
    ).to(device)

    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    best_epoch = ckpt.get("epoch", -1)
    log.info(f"Loaded {best_pt} (epoch {best_epoch})")

    if use_overlap:
        log.info(f"Generating overlap grid (stride={eval_stride}, patch_size={patch_size})...")
        overlap_splits = generate_overlap_splits(
            vacancy_mask_path=vacancy_mask_path,
            borough_mask_path=borough_mask_path,
            split_meta=splits_meta,
            patch_size=patch_size,
            stride=eval_stride,
        )

    all_metrics: dict[str, dict] = {}
    pr_curves: dict[str, np.ndarray] = {}

    for split_name in args.splits:
        if split_name not in splits:
            log.warning(f"Split '{split_name}' not in patch splits, skipping.")
            continue

        if use_overlap:
            overlap_coords = overlap_splits[split_name]
            log.info(f"Evaluating {split_name} with overlap ({len(overlap_coords):,} patches)...")
            m = evaluate_dl_overlap(
                model=model,
                overlap_coords=overlap_coords,
                vrt_path=vrt_path,
                vacancy_mask_path=vacancy_mask_path,
                patch_size=patch_size,
                device=device,
                in_channels=in_channels,
                split_name=f"{arch}/{run_id}/{split_name}",
                building_pred_path=building_pred_path,
                use_building_prob=use_building,
            )
        else:
            ds = NAIPSegmentationDataset(
                vrt_path=vrt_path,
                vacancy_mask_path=vacancy_mask_path,
                patch_coords=splits[split_name],
                patch_size=patch_size,
                in_channels=in_channels,
                building_pred_path=building_pred_path,
                use_building_prob=use_building,
            )
            log.info(f"Evaluating {split_name} ({len(splits[split_name]):,} patches)...")
            m = evaluate_dl_streaming(
                model=model,
                dataset=ds,
                device=device,
                split_name=f"{arch}/{run_id}/{split_name}",
            )

        all_metrics[split_name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()
            if k not in _PR_KEYS
        }
        for k in _PR_KEYS:
            if k in m:
                pr_curves[f"{split_name}_{k}"] = m[k]

    metadata = {
        "run_id": run_id,
        "model_type": arch,
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "decoder_channels": decoder_channels,
        "trained_at": datetime.now().isoformat(),
        "note": note,
        "best_epoch": best_epoch,
        "model": model_cfg_raw,
        "training": training_raw,
        "loss": loss_raw,
        "n_train_patches": len(splits.get("train", [])),
        "eval_stride": eval_stride if use_overlap else patch_size,
        "metrics": all_metrics,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))
    log.info(f"Metrics written to {run_dir / 'metrics.json'}")

    np.savez(run_dir / "pr_curves.npz", **pr_curves)
    log.info(f"PR curves written to {run_dir / 'pr_curves.npz'}")

    log.info("--- Summary ---")
    for split_name, m in all_metrics.items():
        log.info(
            f"  {split_name}: IoU={m['iou']:.4f}, F1={m['f1']:.4f}, "
            f"F2={m['f2']:.4f}, AP={m['average_precision']:.4f}, "
            f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
