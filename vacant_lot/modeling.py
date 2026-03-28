"""
Utilities for supervised classification (parcel-level and pixel-level),
evaluation, and model I/O.
"""
import json
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
)

from .config import CityConfig, DataConfig
from .dataset import compute_spectral_indices
from .logger import get_logger

log = get_logger()


def build_labels(
    df: pd.DataFrame,
    cfg: CityConfig,
) -> pd.Series:
    """
    Create binary vacant/not-vacant labels from a parcel DataFrame.

    Args:
        df: DataFrame containing the land-use column specified in cfg.
        cfg: CityConfig with parcel.landuse_column and parcel.vacant_codes.

    Returns:
        Integer Series (1 = vacant, 0 = not vacant), aligned with df index.
    """
    parcel_cfg = getattr(cfg, "parcel", None) or getattr(cfg, "parcels", None)
    col = parcel_cfg.landuse_column
    codes = parcel_cfg.vacant_codes
    labels = df[col].isin(codes).astype(int)
    n_vacant = labels.sum()
    log.info(
        f"Labels: {n_vacant} vacant ({n_vacant / len(labels) * 100:.1f}%) "
        f"out of {len(labels)} parcels"
    )
    return labels


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract feature matrix and handle NaNs.

    Args:
        df: DataFrame with spectral stats.
        feature_cols: Column names to use as features.

    Returns:
        Tuple of (feature DataFrame with NaNs filled, validated column list).
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].copy()
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        log.info(f"Filling {nan_count} NaN values with column medians")
        X = X.fillna(X.median())

    return X, feature_cols


def evaluate_classifier(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Evaluate a binary classifier and return metrics + PR curve data.

    Args:
        model: Fitted sklearn classifier with predict and predict_proba/decision_function.
        X_val: Validation feature matrix.
        y_val: Validation labels (0/1).
        model_name: Name for logging.

    Returns:
        Dict with keys: f1, average_precision, precision, recall, thresholds,
        classification_report (str).
    """
    y_pred = model.predict(X_val)

    # Get scores for PR curve
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_val)
    else:
        y_scores = y_pred.astype(float)

    f1 = f1_score(y_val, y_pred)
    ap = average_precision_score(y_val, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
    report = classification_report(y_val, y_pred)

    log.info(f"{model_name}: F1={f1:.3f}, AP={ap:.3f}")
    log.info(f"\n{report}")

    return {
        "f1": f1,
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "classification_report": report,
    }


def save_model(
    model,
    path: Path | str,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save a fitted model as .joblib with a JSON metadata sidecar.

    Args:
        model: Fitted sklearn model.
        path: Output path (should end in .joblib).
        metadata: Optional dict of metadata (features, metrics, config info).

    Returns:
        Path to the saved .joblib file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    log.info(f"Saved model to {path}")

    return path


def load_model(path: Path | str) -> tuple:
    """
    Load a model and its metadata sidecar.

    Args:
        path: Path to .joblib file.

    Returns:
        Tuple of (model, metadata_dict). metadata is None if no sidecar exists.
    """
    path = Path(path)
    model = joblib.load(path)
    log.info(f"Loaded model from {path}")

    meta_path = path.with_suffix(".json")
    metadata = None
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        log.info(f"Loaded metadata from {meta_path}")

    return model, metadata


# ---------------------------------------------------------------------------
# Pixel-level segmentation helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "R", "G", "B", "NIR",
    "NDVI", "SAVI", "Brightness", "BareSoilProxy", "EVI", "GNDVI",
]


def extract_patch_features(
    vrt_src: rasterio.DatasetReader,
    mask_src: rasterio.DatasetReader,
    row_off: int,
    col_off: int,
    patch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a single patch and return pixel features, labels, and RGB.

    Mirrors ``NAIPSegmentationDataset.__getitem__`` logic but operates on
    open rasterio handles (no torch dependency).

    Args:
        vrt_src: Open rasterio handle to the NAIP VRT.
        mask_src: Open rasterio handle to the vacancy mask GeoTIFF.
        row_off: Row offset of the patch origin.
        col_off: Column offset of the patch origin.
        patch_size: Side length of the square patch.

    Returns:
        features: (10, H, W) float32 — 4 NAIP bands + 6 spectral indices.
        labels: (H, W) uint8 — 0=non-vacant, 1=vacant, 255=ignore.
        rgb: (3, H, W) uint8 — raw R, G, B bands for visualization.
    """
    win = Window(col_off, row_off, patch_size, patch_size)

    rgbn = vrt_src.read(window=win)  # (4, H, W) uint8
    labels = mask_src.read(1, window=win)  # (H, W) uint8

    indices = compute_spectral_indices(rgbn)  # (6, H, W) float32
    naip_scaled = rgbn.astype(np.float32) / 255.0  # (4, H, W) float32
    features = np.concatenate([naip_scaled, indices], axis=0)  # (10, H, W)

    return features, labels, rgbn[:3]


def sample_pixels_from_patches(
    vrt_path: Path | str,
    mask_path: Path | str,
    patch_coords: list[tuple[int, int]],
    n_vacant: int = 2_000_000,
    n_nonvacant: int = 6_000_000,
    patch_size: int = 256,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pixel-level features + labels with stratified reservoir sampling.

    Iterates all patches in a single pass, maintaining two fixed-size
    reservoirs (one per class). Ignore pixels (label=255) are skipped.

    Args:
        vrt_path: Path to NAIP VRT.
        mask_path: Path to vacancy mask GeoTIFF.
        patch_coords: List of (row_offset, col_offset) tuples.
        n_vacant: Maximum number of vacant pixels to sample.
        n_nonvacant: Maximum number of non-vacant pixels to sample.
        patch_size: Side length of square patches.
        random_state: Seed for reproducibility.

    Returns:
        X: (N, 10) float32 feature matrix.
        y: (N,) int8 label array (0 or 1).
    """
    rng = np.random.default_rng(random_state)

    # Pre-allocate reservoirs
    res_vac_X = np.empty((n_vacant, 10), dtype=np.float32)
    res_vac_count = 0
    res_nv_X = np.empty((n_nonvacant, 10), dtype=np.float32)
    res_nv_count = 0

    # Running count of total pixels seen per class (for reservoir probability)
    total_vac_seen = 0
    total_nv_seen = 0

    def _reservoir_insert_batch(
        reservoir: np.ndarray,
        res_count: int,
        res_capacity: int,
        total_seen: int,
        batch: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Vectorized reservoir sampling for a batch of items."""
        n = len(batch)
        if n == 0:
            return res_count, total_seen

        # Phase 1: fill reservoir if not full
        if res_count < res_capacity:
            space = min(n, res_capacity - res_count)
            reservoir[res_count : res_count + space] = batch[:space]
            res_count += space
            total_seen += space
            batch = batch[space:]
            n = len(batch)
            if n == 0:
                return res_count, total_seen

        # Phase 2: reservoir is full, use standard algorithm R
        # For each item at index total_seen+k, accept with prob capacity/(total_seen+k+1)
        indices = np.arange(total_seen, total_seen + n)
        # Generate random integers in [0, index] for each item
        rand_vals = rng.integers(0, indices + 1)
        # Accept if rand_val < capacity
        accept_mask = rand_vals < res_capacity
        if accept_mask.any():
            positions = rand_vals[accept_mask]
            reservoir[positions] = batch[accept_mask]

        total_seen += n
        return res_count, total_seen

    from tqdm import tqdm

    with rasterio.open(vrt_path) as vrt_src, rasterio.open(mask_path) as mask_src:
        for i, (row_off, col_off) in enumerate(tqdm(patch_coords, desc="Sampling", unit="patch")):
            features, labels, _ = extract_patch_features(
                vrt_src, mask_src, row_off, col_off, patch_size
            )

            # Reshape to (N_pixels, 10) and flatten labels
            feat_flat = features.reshape(10, -1).T  # (H*W, 10)
            lab_flat = labels.ravel()  # (H*W,)

            # Separate by class, skip ignore
            vac_pixels = feat_flat[lab_flat == 1]
            nv_pixels = feat_flat[lab_flat == 0]

            res_vac_count, total_vac_seen = _reservoir_insert_batch(
                res_vac_X, res_vac_count, n_vacant, total_vac_seen, vac_pixels, rng
            )
            res_nv_count, total_nv_seen = _reservoir_insert_batch(
                res_nv_X, res_nv_count, n_nonvacant, total_nv_seen, nv_pixels, rng
            )


    # Trim to actual counts
    res_vac_X = res_vac_X[:res_vac_count]
    res_nv_X = res_nv_X[:res_nv_count]

    X = np.concatenate([res_vac_X, res_nv_X], axis=0)
    y = np.concatenate([
        np.ones(res_vac_count, dtype=np.int8),
        np.zeros(res_nv_count, dtype=np.int8),
    ])

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    log.info(
        f"Sampled {len(y):,} pixels: "
        f"{res_vac_count:,} vacant (of {total_vac_seen:,}), "
        f"{res_nv_count:,} non-vacant (of {total_nv_seen:,})"
    )
    return X, y


def evaluate_segmentation_streaming(
    model,
    vrt_path: Path | str,
    mask_path: Path | str,
    patch_coords: list[tuple[int, int]],
    patch_size: int = 256,
    reservoir_size: int = 100_000,
    random_state: int = 42,
    model_name: str = "model",
) -> dict:
    """
    Stream-evaluate a pixel classifier over patches, accumulating metrics.

    Iterates patches one-by-one, predicts per-pixel, accumulates a confusion
    matrix (TP/FP/FN/TN) and a reservoir sample of (score, label) pairs for
    PR curve / AP computation. O(1) memory beyond the model itself.

    Args:
        model: Fitted sklearn classifier with predict() and predict_proba().
        vrt_path: Path to NAIP VRT.
        mask_path: Path to vacancy mask GeoTIFF.
        patch_coords: Patch coordinates for the split to evaluate.
        patch_size: Side length of square patches.
        reservoir_size: Number of (score, label) pairs to keep for AP.
        random_state: Seed for reservoir sampling.
        model_name: Name for logging.

    Returns:
        Dict with: iou, dice, precision, recall, f1, kappa,
        average_precision, confusion_matrix (2x2 ndarray),
        pr_precision, pr_recall, pr_thresholds (arrays for PR curve),
        n_pixels_evaluated.
    """
    rng = np.random.default_rng(random_state)

    # Confusion matrix accumulators
    tp = fn = fp = tn = 0

    # Reservoir for AP computation (stratified: half pos, half neg)
    half = reservoir_size // 2
    res_pos_scores = np.empty(half, dtype=np.float32)
    res_pos_count = 0
    total_pos_seen = 0
    res_neg_scores = np.empty(half, dtype=np.float32)
    res_neg_count = 0
    total_neg_seen = 0

    from tqdm import tqdm

    with rasterio.open(vrt_path) as vrt_src, rasterio.open(mask_path) as mask_src:
        for i, (row_off, col_off) in enumerate(tqdm(patch_coords, desc=f"Eval {model_name}", unit="patch")):
            features, labels, _ = extract_patch_features(
                vrt_src, mask_src, row_off, col_off, patch_size
            )

            feat_flat = features.reshape(10, -1).T  # (H*W, 10)
            lab_flat = labels.ravel()

            # Only evaluate on labeled pixels
            valid = lab_flat != 255
            if not valid.any():
                continue

            X_patch = feat_flat[valid]
            y_true = lab_flat[valid]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                y_pred = model.predict(X_patch)
                scores = model.predict_proba(X_patch)[:, 1]

            # Accumulate confusion matrix
            pos = y_true == 1
            neg = ~pos
            tp += int((y_pred[pos] == 1).sum())
            fn += int((y_pred[pos] == 0).sum())
            fp += int((y_pred[neg] == 1).sum())
            tn += int((y_pred[neg] == 0).sum())

            # Reservoir sample scores for AP (stratified, vectorized)
            pos_scores_batch = scores[pos]
            n_ps = len(pos_scores_batch)
            if n_ps > 0:
                if res_pos_count < half:
                    space = min(n_ps, half - res_pos_count)
                    res_pos_scores[res_pos_count : res_pos_count + space] = pos_scores_batch[:space]
                    res_pos_count += space
                    total_pos_seen += space
                    pos_scores_batch = pos_scores_batch[space:]
                    n_ps = len(pos_scores_batch)
                if n_ps > 0:
                    indices = np.arange(total_pos_seen, total_pos_seen + n_ps)
                    rand_vals = rng.integers(0, indices + 1)
                    accept = rand_vals < half
                    if accept.any():
                        res_pos_scores[rand_vals[accept]] = pos_scores_batch[accept]
                    total_pos_seen += n_ps

            neg_scores_batch = scores[neg]
            n_ns = len(neg_scores_batch)
            if n_ns > 0:
                if res_neg_count < half:
                    space = min(n_ns, half - res_neg_count)
                    res_neg_scores[res_neg_count : res_neg_count + space] = neg_scores_batch[:space]
                    res_neg_count += space
                    total_neg_seen += space
                    neg_scores_batch = neg_scores_batch[space:]
                    n_ns = len(neg_scores_batch)
                if n_ns > 0:
                    indices = np.arange(total_neg_seen, total_neg_seen + n_ns)
                    rand_vals = rng.integers(0, indices + 1)
                    accept = rand_vals < half
                    if accept.any():
                        res_neg_scores[rand_vals[accept]] = neg_scores_batch[accept]
                    total_neg_seen += n_ns


    # Compute metrics from confusion matrix
    n_pixels = tp + fp + fn + tn
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1_val = 2 * prec * rec / max(prec + rec, 1e-8)
    f2_val = 5 * prec * rec / max(4 * prec + rec, 1e-8)

    # Cohen's kappa from confusion matrix
    total = max(n_pixels, 1)
    p_o = (tp + tn) / total
    p_pos = ((tp + fp) * (tp + fn)) / (total * total)
    p_neg = ((tn + fn) * (tn + fp)) / (total * total)
    p_e = p_pos + p_neg
    kappa = (p_o - p_e) / max(1 - p_e, 1e-8)

    # Compute AP and PR curve from reservoir samples
    res_scores = np.concatenate([
        res_pos_scores[:res_pos_count],
        res_neg_scores[:res_neg_count],
    ])
    res_labels = np.concatenate([
        np.ones(res_pos_count, dtype=np.int8),
        np.zeros(res_neg_count, dtype=np.int8),
    ])

    if len(res_scores) > 0 and res_pos_count > 0:
        # Weight samples to reflect true class frequencies
        pos_weight = total_pos_seen / max(res_pos_count, 1)
        neg_weight = total_neg_seen / max(res_neg_count, 1)
        sample_weights = np.where(res_labels == 1, pos_weight, neg_weight)
        ap = average_precision_score(res_labels, res_scores, sample_weight=sample_weights)
        pr_prec, pr_rec, pr_thresh = precision_recall_curve(
            res_labels, res_scores, sample_weight=sample_weights
        )
    else:
        ap = 0.0
        pr_prec, pr_rec, pr_thresh = np.array([0.0]), np.array([0.0]), np.array([0.0])

    cm = np.array([[tn, fp], [fn, tp]])

    log.info(
        f"{model_name}: IoU={iou:.4f}, Dice={dice:.4f}, "
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


def predict_patch(
    model,
    vrt_src: rasterio.DatasetReader,
    mask_src: rasterio.DatasetReader,
    row_off: int,
    col_off: int,
    patch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict on a single patch for visualization.

    Args:
        model: Fitted sklearn classifier with predict() and predict_proba().
        vrt_src: Open rasterio handle to the NAIP VRT.
        mask_src: Open rasterio handle to the vacancy mask GeoTIFF.
        row_off: Row offset of the patch origin.
        col_off: Column offset of the patch origin.
        patch_size: Side length of the square patch.

    Returns:
        pred_mask: (H, W) uint8 — 0/1/255 (255 copied from true mask).
        true_mask: (H, W) uint8 — ground truth labels.
        prob_map: (H, W) float32 — P(vacant) from predict_proba.
        rgb: (3, H, W) uint8 — raw RGB bands.
    """
    features, true_mask, rgb = extract_patch_features(
        vrt_src, mask_src, row_off, col_off, patch_size
    )

    feat_flat = features.reshape(10, -1).T  # (H*W, 10)
    lab_flat = true_mask.ravel()
    valid = lab_flat != 255

    pred_flat = np.full(len(lab_flat), 255, dtype=np.uint8)
    prob_flat = np.full(len(lab_flat), np.nan, dtype=np.float32)

    if valid.any():
        pred_flat[valid] = model.predict(feat_flat[valid]).astype(np.uint8)
        prob_flat[valid] = model.predict_proba(feat_flat[valid])[:, 1]

    h, w = true_mask.shape
    pred_mask = pred_flat.reshape(h, w)
    prob_map = prob_flat.reshape(h, w)

    return pred_mask, true_mask, prob_map, rgb
