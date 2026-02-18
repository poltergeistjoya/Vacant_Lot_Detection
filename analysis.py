"""
Analysis utilities for spectral clustering and feature extraction.
"""
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from logger import get_logger

log = get_logger()


# ============================================================================
# GCS Data Loading
# ============================================================================

def read_csv_from_gcs(
    bucket_name: str,
    blob_path: str,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file from Google Cloud Storage into a DataFrame.

    Args:
        bucket_name: GCS bucket name (e.g., 'thesis_parcels').
        blob_path: Path to the file within the bucket (e.g., 'eda/new_york_new_york/stats.csv').
        **read_csv_kwargs: Additional arguments passed to pd.read_csv().

    Returns:
        DataFrame with the CSV contents.

    Example:
        df = read_csv_from_gcs(
            bucket_name='thesis_parcels',
            blob_path='eda/new_york_new_york/parcel_spectral_stats.csv'
        )
    """
    from google.cloud import storage

    log.info(f"Reading CSV from gs://{bucket_name}/{blob_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    csv_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(csv_bytes), **read_csv_kwargs)

    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_spectral_stats(
    bucket_name: str,
    blob_path: str,
    parcel_df: Optional[pd.DataFrame] = None,
    parcel_id_col: str = "BBL",
    join_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load spectral stats from GCS and optionally join with parcel metadata.

    Args:
        bucket_name: GCS bucket name.
        blob_path: Path to spectral stats CSV in bucket.
        parcel_df: Optional DataFrame with parcel metadata (e.g., LandUse).
        parcel_id_col: Column name for joining (default: 'BBL').
        join_cols: Columns to keep from parcel_df. If None, keeps all.

    Returns:
        DataFrame with spectral stats, optionally merged with parcel metadata.

    Example:
        # Load and join with parcel data
        df = load_spectral_stats(
            bucket_name='thesis_parcels',
            blob_path='eda/new_york_new_york/parcel_spectral_stats.csv',
            parcel_df=sampled_gdf,
            join_cols=['BBL', 'LandUse', 'Shape_Area']
        )
    """
    # Load spectral stats from GCS
    spectral_df = read_csv_from_gcs(bucket_name, blob_path)

    if parcel_df is None:
        return spectral_df

    # Prepare parcel data for join
    if join_cols is not None:
        # Ensure parcel_id_col is included
        if parcel_id_col not in join_cols:
            join_cols = [parcel_id_col] + join_cols
        parcel_subset = parcel_df[join_cols].copy()
    else:
        parcel_subset = parcel_df.copy()

    # Merge on parcel ID
    log.info(f"Joining spectral stats with parcel data on '{parcel_id_col}'")
    merged = spectral_df.merge(parcel_subset, on=parcel_id_col, how="left")

    log.info(f"Merged result: {len(merged)} rows")
    return merged


# ============================================================================
# General Clustering Pipeline
# ============================================================================

def cluster_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int = 5,
    random_state: int = 42,
    add_pca: bool = True,
    n_pca_components: int = 2,
    cluster_col_name: str = "cluster",
    fill_na: str = "median",
) -> tuple[pd.DataFrame, dict]:
    """
    General-purpose clustering pipeline for any DataFrame.

    This function can be used for spectral features, geometric features,
    or any combination of numeric features.

    Args:
        df: Input DataFrame with features.
        feature_columns: List of column names to use as clustering features.
        n_clusters: Number of clusters for KMeans.
        random_state: Random seed for reproducibility.
        add_pca: Whether to add PCA components for visualization.
        n_pca_components: Number of PCA components (default: 2).
        cluster_col_name: Name for the cluster label column (default: 'cluster').
        fill_na: How to fill NaN values ('median', 'mean', or 'drop').

    Returns:
        Tuple of:
        - DataFrame with added cluster labels and optionally PCA components
        - Dict with fitted models {'scaler': StandardScaler, 'kmeans': KMeans, 'pca': PCA}

    Example:
        # Cluster on spectral features
        df, models = cluster_dataframe(
            df=spectral_df,
            feature_columns=['R_mean', 'G_mean', 'B_mean', 'NDVI_mean'],
            n_clusters=5
        )

        # Cluster on geometric + spectral features
        df, models = cluster_dataframe(
            df=merged_df,
            feature_columns=['Shape_Area', 'geom_perimeter', 'NDVI_mean', 'Brightness_mean'],
            n_clusters=8
        )
    """
    log.info(f"Starting clustering pipeline with {len(feature_columns)} features")

    # Validate feature columns exist
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Extract features
    X = df[feature_columns].copy()
    log.info(f"Features: {feature_columns}")

    # Handle NaN values
    nan_counts = X.isna().sum().sum()
    if nan_counts > 0:
        log.info(f"Found {nan_counts} NaN values")
        if fill_na == "median":
            X = X.fillna(X.median())
        elif fill_na == "mean":
            X = X.fillna(X.mean())
        elif fill_na == "drop":
            X = X.dropna()
            log.info(f"Dropped rows with NaN, {len(X)} rows remaining")
        else:
            raise ValueError(f"Unknown fill_na method: {fill_na}")

    # Scale features
    log.info("Scaling features with StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run KMeans
    log.info(f"Running KMeans with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Log cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        log.info(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")

    # Build result DataFrame
    result = df.copy()
    if fill_na == "drop":
        result = result.loc[X.index]
    result[cluster_col_name] = labels

    # Store fitted models
    models = {
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_columns": feature_columns,
    }

    # Add PCA for visualization
    if add_pca:
        log.info(f"Running PCA with {n_pca_components} components")
        pca = PCA(n_components=n_pca_components, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)

        for i in range(n_pca_components):
            result[f"pc{i+1}"] = X_pca[:, i]

        # Log explained variance
        for i, var in enumerate(pca.explained_variance_ratio_):
            log.info(f"  PC{i+1} explained variance: {var*100:.1f}%")
        log.info(f"  Total explained variance: {sum(pca.explained_variance_ratio_)*100:.1f}%")

        models["pca"] = pca

    log.info("Clustering pipeline complete")
    return result, models


def predict_clusters(
    df: pd.DataFrame,
    models: dict,
    add_pca: bool = True,
    cluster_col_name: str = "cluster",
) -> pd.DataFrame:
    """
    Predict cluster labels for new data using fitted models.

    Args:
        df: New DataFrame with same feature columns as training data.
        models: Dict with fitted models from cluster_dataframe().
        add_pca: Whether to add PCA components.
        cluster_col_name: Name for the cluster label column.

    Returns:
        DataFrame with cluster labels and optionally PCA components.
    """
    feature_columns = models["feature_columns"]
    scaler = models["scaler"]
    kmeans = models["kmeans"]

    # Extract and scale features
    X = df[feature_columns].copy()
    X = X.fillna(X.median())  # Use same fill strategy
    X_scaled = scaler.transform(X)

    # Predict clusters
    labels = kmeans.predict(X_scaled)

    result = df.copy()
    result[cluster_col_name] = labels

    if add_pca and "pca" in models:
        pca = models["pca"]
        X_pca = pca.transform(X_scaled)
        for i in range(X_pca.shape[1]):
            result[f"pc{i+1}"] = X_pca[:, i]

    return result


def prepare_spectral_features(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    suffix: str = "_mean",
) -> pd.DataFrame:
    """
    Extract and prepare spectral features for clustering.

    Args:
        df: DataFrame with spectral stats per parcel.
        feature_columns: Specific columns to use. If None, uses all columns
                        ending with `suffix`.
        suffix: Column suffix to filter by (default: '_mean').

    Returns:
        DataFrame with only the feature columns, NaN values filled with median.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c.endswith(suffix)]
        log.info(f"Auto-selected {len(feature_columns)} features with suffix '{suffix}'")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_columns].copy()

    # Fill NaN with column median
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        log.info(f"Filling {nan_counts.sum()} NaN values with column medians")
        X = X.fillna(X.median())

    return X


def scale_features(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Scale features to zero mean and unit variance.

    Args:
        X: DataFrame with features.
        scaler: Optional pre-fitted scaler. If None, fits a new one.

    Returns:
        Tuple of (scaled_array, fitted_scaler).
    """
    if scaler is None:
        log.info("Fitting new StandardScaler")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        log.info("Using pre-fitted scaler")
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def run_kmeans_clustering(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """
    Run KMeans clustering on scaled features.

    Args:
        X_scaled: Scaled feature array.
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (cluster_labels, fitted_kmeans_model).
    """
    log.info(f"Running KMeans with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Log cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        log.info(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")

    return labels, kmeans


def run_pca(
    X_scaled: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Run PCA for dimensionality reduction / visualization.

    Args:
        X_scaled: Scaled feature array.
        n_components: Number of principal components.
        random_state: Random seed.

    Returns:
        Tuple of (transformed_array, fitted_pca_model).
    """
    log.info(f"Running PCA with {n_components} components")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Log explained variance
    for i, var in enumerate(pca.explained_variance_ratio_):
        log.info(f"  PC{i+1} explained variance: {var*100:.1f}%")
    log.info(f"  Total explained variance: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    return X_pca, pca


def cluster_spectral_data(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    n_clusters: int = 5,
    random_state: int = 42,
    add_pca: bool = True,
) -> pd.DataFrame:
    """
    Full clustering pipeline: prepare features, scale, cluster, and add PCA.

    Args:
        df: DataFrame with spectral stats per parcel.
        feature_columns: Columns to use as features. If None, auto-selects *_mean columns.
        n_clusters: Number of clusters for KMeans.
        random_state: Random seed.
        add_pca: Whether to add PC1, PC2 columns for visualization.

    Returns:
        DataFrame with added 'cluster', and optionally 'pc1', 'pc2' columns.
    """
    log.info("Starting spectral clustering pipeline")

    # Prepare features
    X = prepare_spectral_features(df, feature_columns)
    feature_names = list(X.columns)
    log.info(f"Features: {feature_names}")

    # Scale
    X_scaled, scaler = scale_features(X)

    # Cluster
    labels, kmeans = run_kmeans_clustering(X_scaled, n_clusters, random_state)

    # Add results to dataframe
    result = df.copy()
    result["cluster"] = labels

    if add_pca:
        X_pca, pca = run_pca(X_scaled, n_components=2, random_state=random_state)
        result["pc1"] = X_pca[:, 0]
        result["pc2"] = X_pca[:, 1]

    log.info("Clustering pipeline complete")
    return result


def analyze_cluster_composition(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    category_col: str = "LandUse",
) -> pd.DataFrame:
    """
    Analyze the composition of each cluster by a categorical variable.

    Args:
        df: DataFrame with cluster assignments.
        cluster_col: Name of the cluster column.
        category_col: Name of the category column (e.g., LandUse).

    Returns:
        DataFrame with category proportions per cluster (unstacked).
    """
    log.info(f"Analyzing cluster composition by {category_col}")

    composition = (
        df.groupby(cluster_col)[category_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    return composition


def get_cluster_feature_stats(
    df: pd.DataFrame,
    feature_columns: list[str],
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Compute mean feature values per cluster.

    Args:
        df: DataFrame with cluster assignments and features.
        feature_columns: List of feature column names.
        cluster_col: Name of the cluster column.

    Returns:
        DataFrame with mean values per cluster.
    """
    stats = df.groupby(cluster_col)[feature_columns].mean()
    return stats


def find_optimal_clusters(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate clustering quality across different k values using inertia and silhouette.

    Args:
        X_scaled: Scaled feature array.
        k_range: Range of cluster counts to evaluate.
        random_state: Random seed.

    Returns:
        DataFrame with k, inertia, and silhouette_score columns.
    """
    from sklearn.metrics import silhouette_score

    log.info(f"Evaluating clusters for k in {list(k_range)}")

    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette_score": sil_score,
        })
        log.info(f"  k={k}: inertia={kmeans.inertia_:.1f}, silhouette={sil_score:.3f}")

    return pd.DataFrame(results)
