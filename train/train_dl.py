"""
Train a deep learning segmentation model (UNet or DeepLabV3+) on NAIP imagery.

Builds a model via segmentation-models-pytorch, trains with BCE+Dice loss and
early stopping, then post-training evaluates best.pt on val and test splits.

Usage:
  uv run python train/train_dl.py --config unet_32.yaml
  uv run python train/train_dl.py --config unet_resnet18.yaml --run-id 003
  uv run python train/train_dl.py --config deeplabv3_resnet18.yaml --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from vacant_lot.config import DLTrainConfig, load_train_config, _get_shared_root
from vacant_lot.dataset import NAIPSegmentationDataset, load_patch_splits
from vacant_lot.logger import get_logger
from vacant_lot.train import SegmentationTrainer, _auto_device

log = get_logger()


# ---------------------------------------------------------------------------
# Run ID helpers
# ---------------------------------------------------------------------------

def next_run_id(output_dir: Path) -> str:
    """Return the next zero-padded 3-digit run ID by scanning output_dir."""
    if not output_dir.exists():
        return "001"
    existing = sorted(
        int(p.name) for p in output_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    return f"{(existing[-1] + 1 if existing else 1):03d}"


# ---------------------------------------------------------------------------
# DL streaming evaluation
# ---------------------------------------------------------------------------

def evaluate_dl_streaming(
    model: torch.nn.Module,
    dataset: NAIPSegmentationDataset,
    device: torch.device,
    reservoir_size: int = 100_000,
    random_state: int = 42,
    split_name: str = "val",
) -> dict:
    """Evaluate a DL segmentation model patch-by-patch.

    Iterates over individual patches (batch_size=1) to keep memory bounded.
    Accumulates a confusion matrix and a reservoir-sampled set of (score, label)
    pairs for AP / PR curve computation.

    Returns a dict with the same schema as evaluate_segmentation_streaming:
        iou, dice, precision, recall, f1, f2, kappa, average_precision,
        confusion_matrix (2×2 ndarray), pr_precision, pr_recall,
        pr_thresholds, n_pixels_evaluated.
    """
    model.eval()
    rng = np.random.default_rng(random_state)

    tp = fp = fn = tn = 0

    half = reservoir_size // 2
    res_pos_scores = np.empty(half, dtype=np.float32)
    res_neg_scores = np.empty(half, dtype=np.float32)
    res_pos_count = res_neg_count = 0
    total_pos_seen = total_neg_seen = 0

    # Use a single-worker loader to avoid rasterio/GDAL fork-safety issues on macOS
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Eval {split_name}", unit="patch"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)                    # (1, 1, H, W)
            probs = torch.sigmoid(logits).squeeze()   # (H, W)
            mask_2d = masks.squeeze()                 # (H, W)

            valid = mask_2d != 255
            if not valid.any():
                continue

            scores_valid = probs[valid].cpu().numpy().astype(np.float32)
            labels_valid = (mask_2d[valid] == 1).cpu().numpy()

            pred_bin = scores_valid > 0.5
            pos = labels_valid == 1
            neg = ~pos

            tp += int((pred_bin[pos]).sum())
            fn += int((~pred_bin[pos]).sum())
            fp += int((pred_bin[neg]).sum())
            tn += int((~pred_bin[neg]).sum())

            # Reservoir sampling for AP (stratified)
            def _reservoir_update(batch_scores, res_buf, res_count, total_seen):
                n = len(batch_scores)
                if n == 0:
                    return res_count, total_seen
                if res_count < half:
                    space = min(n, half - res_count)
                    res_buf[res_count : res_count + space] = batch_scores[:space]
                    res_count += space
                    total_seen += space
                    batch_scores = batch_scores[space:]
                    n = len(batch_scores)
                if n > 0:
                    indices = np.arange(total_seen, total_seen + n)
                    rand_vals = rng.integers(0, indices + 1)
                    accept = rand_vals < half
                    if accept.any():
                        res_buf[rand_vals[accept]] = batch_scores[accept]
                    total_seen += n
                return res_count, total_seen

            res_pos_count, total_pos_seen = _reservoir_update(
                scores_valid[pos], res_pos_scores, res_pos_count, total_pos_seen
            )
            res_neg_count, total_neg_seen = _reservoir_update(
                scores_valid[neg], res_neg_scores, res_neg_count, total_neg_seen
            )

    # Compute metrics from confusion matrix
    n_pixels = tp + fp + fn + tn
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1_val = 2 * prec * rec / max(prec + rec, 1e-8)
    f2_val = 5 * prec * rec / max(4 * prec + rec, 1e-8)

    total = max(n_pixels, 1)
    p_o = (tp + tn) / total
    p_pos = ((tp + fp) * (tp + fn)) / (total * total)
    p_neg = ((tn + fn) * (tn + fp)) / (total * total)
    p_e = p_pos + p_neg
    kappa = (p_o - p_e) / max(1 - p_e, 1e-8)

    res_scores = np.concatenate([
        res_pos_scores[:res_pos_count],
        res_neg_scores[:res_neg_count],
    ])
    res_labels = np.concatenate([
        np.ones(res_pos_count, dtype=np.int8),
        np.zeros(res_neg_count, dtype=np.int8),
    ])

    if len(res_scores) > 0 and res_pos_count > 0:
        pos_weight = total_pos_seen / max(res_pos_count, 1)
        neg_weight = total_neg_seen / max(res_neg_count, 1)
        sample_weights = np.where(res_labels == 1, pos_weight, neg_weight)
        ap = average_precision_score(res_labels, res_scores, sample_weight=sample_weights)
        pr_prec, pr_rec, pr_thresh = precision_recall_curve(
            res_labels, res_scores, sample_weight=sample_weights
        )
    else:
        ap = 0.0
        pr_prec = pr_rec = pr_thresh = np.array([0.0])

    cm = np.array([[tn, fp], [fn, tp]])

    log.info(
        f"{split_name}: IoU={iou:.4f}, Dice={dice:.4f}, "
        f"Prec={prec:.4f}, Rec={rec:.4f}, F1={f1_val:.4f}, F2={f2_val:.4f}, "
        f"Kappa={kappa:.4f}, AP={ap:.4f} | {n_pixels:,} pixels"
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": prec,
        "recall": rec,
        "f1": f1_val,
        "f2": f2_val,
        "kappa": kappa,
        "average_precision": ap,
        "confusion_matrix": cm,
        "pr_precision": pr_prec,
        "pr_recall": pr_rec,
        "pr_thresholds": pr_thresh,
        "n_pixels_evaluated": n_pixels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DL segmentation model (UNet / DeepLabV3+)")
    parser.add_argument(
        "--config",
        default="unet_32.yaml",
        help="Path to DL training config YAML (default: config/unet_32.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (default: auto-increment from output_dir)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest.pt if it exists in the run directory",
    )
    args = parser.parse_args()

    data_cfg, train_cfg = load_train_config(args.config)

    if not isinstance(train_cfg, DLTrainConfig):
        log.error(f"Expected DLTrainConfig, got {type(train_cfg).__name__}. Use a DL config YAML.")
        sys.exit(1)

    model_cfg = train_cfg.model
    training_cfg = train_cfg.training
    loss_cfg = train_cfg.loss
    shared_root = data_cfg._shared_root
    note = os.environ.get("VACANT_LOT_RUN_NOTE", train_cfg.note)

    # Disable MPS memory pool — forces immediate deallocation of Metal buffers
    # instead of caching them. Without this, Activity Monitor shows unbounded
    # memory growth even though Python's RSS is stable (the leak is in Metal's
    # unified memory allocator). Slightly slower but prevents OOM.
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Cap GDAL block cache to 256MB to prevent unbounded tile caching
    # (default is 5% of RAM, but GDAL doesn't always respect the cap)
    try:
        from osgeo import gdal
        gdal.SetCacheMax(256 * 1024 * 1024)
    except ImportError:
        pass

    # Seed for reproducibility
    seed = training_cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    base_output_dir = shared_root / train_cfg.output_dir
    run_id = args.run_id or next_run_id(base_output_dir)
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = shared_root / "outputs" / "logs" / model_cfg.type / run_id

    log.info(f"Run:           {model_cfg.type}/{run_id}")
    log.info(f"Output dir:    {run_dir}")
    log.info(f"Log dir:       {log_dir}")
    log.info(f"Config:        {args.config}")
    if note:
        log.info(f"Note:          {note}")

    # Save a copy of the config YAML into the run directory for reproducibility.
    import shutil
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config_src = config_dir / args.config
    if config_src.exists():
        shutil.copy2(config_src, run_dir / "config.yaml")

    vrt_path = data_cfg.get_vrt_path()
    vacancy_mask_path = data_cfg.get_vacancy_mask_path()
    splits_path = data_cfg.get_patch_splits_path()

    splits = load_patch_splits(splits_path)
    patch_size = data_cfg.patch.size

    # DataLoaders
    # num_workers=0 on macOS: rasterio uses GDAL which is not fork-safe.
    train_dataset = NAIPSegmentationDataset(
        vrt_path=vrt_path,
        vacancy_mask_path=vacancy_mask_path,
        patch_coords=splits["train"],
        patch_size=patch_size,
    )
    val_dataset = NAIPSegmentationDataset(
        vrt_path=vrt_path,
        vacancy_mask_path=vacancy_mask_path,
        patch_coords=splits["val"],
        patch_size=patch_size,
    )

    n_workers = training_cfg.num_workers
    use_cuda = torch.cuda.is_available()
    # On CUDA/Linux, use multiple workers + pin_memory for faster host→device transfer.
    # On MPS/macOS, keep num_workers=0 to avoid GDAL fork-safety issues and memory leaks.
    if use_cuda:
        pin_memory = True
    else:
        n_workers = 0
        pin_memory = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=n_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.batch_size * 4,  # no gradients, larger batch is fine
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=n_workers > 0,
    )

    device = _auto_device()
    log.info(f"Device:        {device}")

    trainer = SegmentationTrainer(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        loss_cfg=loss_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=run_dir,
        log_dir=log_dir,
        device=device,
        run_id=run_id,
    )

    if args.resume:
        latest_pt = run_dir / "latest.pt"
        if latest_pt.exists():
            trainer.load_checkpoint(latest_pt)
        else:
            log.warning(f"--resume specified but {latest_pt} not found, starting fresh.")

    best_metrics, history = trainer.train()

    # Save training history
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info(f"History written to {run_dir / 'history.json'}")

    # -------------------------------------------------------------------
    # Post-training eval using best.pt
    # -------------------------------------------------------------------
    best_pt = run_dir / "best.pt"
    if not best_pt.exists():
        log.error(f"best.pt not found at {best_pt}, skipping post-training eval.")
        sys.exit(1)

    log.info("Loading best.pt for final evaluation...")
    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = best_metrics.get("best_epoch", ckpt.get("epoch", -1))

    _PR_KEYS = ("pr_precision", "pr_recall", "pr_thresholds")
    all_metrics: dict[str, dict] = {}
    pr_curves: dict[str, np.ndarray] = {}

    for split_name in ("val", "test"):
        split_coords = splits[split_name]
        ds = val_dataset if split_name == "val" else NAIPSegmentationDataset(
            vrt_path=vrt_path,
            vacancy_mask_path=vacancy_mask_path,
            patch_coords=split_coords,
            patch_size=patch_size,
        )
        log.info(f"Evaluating on {split_name} ({len(split_coords):,} patches)...")
        m = evaluate_dl_streaming(
            model=trainer.model,
            dataset=ds,
            device=device,
            split_name=f"{model_cfg.type}/{run_id}/{split_name}",
        )
        all_metrics[split_name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()
            if k not in _PR_KEYS
        }
        for k in _PR_KEYS:
            if k in m:
                pr_curves[f"{split_name}_{k}"] = m[k]

    # Save metrics.json — reflects best model performance
    metadata = {
        "run_id": run_id,
        "model_type": model_cfg.type,
        "encoder_name": model_cfg.encoder_name,
        "encoder_weights": model_cfg.encoder_weights,
        "decoder_channels": model_cfg.decoder_channels if hasattr(model_cfg, "decoder_channels") else None,
        "trained_at": datetime.now().isoformat(),
        "config_file": args.config,
        "note": note,
        "best_epoch": best_epoch,
        "model": model_cfg.model_dump(),
        "training": training_cfg.model_dump(),
        "loss": loss_cfg.model_dump(),
        "n_train_patches": len(splits["train"]),
        "metrics": all_metrics,
    }

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
