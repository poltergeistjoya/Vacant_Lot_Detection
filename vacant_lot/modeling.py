"""
Utilities for supervised parcel classification (label building, evaluation, model I/O).
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)

from .config import CityConfig
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
    col = cfg.parcel.landuse_column
    codes = cfg.parcel.vacant_codes
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

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        # Convert non-serializable values
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v
        meta_path.write_text(json.dumps(clean, indent=2))
        log.info(f"Saved metadata to {meta_path}")

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
