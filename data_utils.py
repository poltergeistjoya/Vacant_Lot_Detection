from pathlib import Path
import fiona
import geopandas as gpd
import pandas as pd
from joblib import Memory
from typing import List, Dict
from logger import get_logger


def upload_to_gcs(
    local_path: Path | str,
    bucket_name: str,
    gcs_prefix: str,
    gcs_filename: str,
) -> str:
    """
    Upload a local file to Google Cloud Storage.

    Args:
        local_path: Local path to the file to upload.
        bucket_name: GCS bucket name.
        gcs_prefix: Path prefix in bucket (e.g., 'eda/new_york_new_york').
        gcs_filename: Filename in GCS.

    Returns:
        Full GCS URI (gs://bucket/prefix/filename).
    """
    from google.cloud import storage

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    gcs_path = f"{gcs_prefix}/{gcs_filename}"
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(gcs_path)

    log.info(f"Uploading {local_path} to gs://{bucket_name}/{gcs_path}")
    blob.upload_from_filename(local_path)
    log.info(f"Upload complete: gs://{bucket_name}/{gcs_path}")

    return f"gs://{bucket_name}/{gcs_path}"

log = get_logger()
memory = Memory("cache", verbose=0)

@memory.cache
def load_gdb(gdb_path:str | Path, layer:str | None = None, subset_size:int | None = None, random_state:int =42):
    """
    Load .gdb
    
    Args:
        gdb_path (str | Path): Path to the .gdb file.
        layer (str | None): Optional layer name to load.
        subset_size (int | None): Optional sample size for quick testing.

    Returns:
        geopandas.GeoDataFrame
    """

    # make path absolute 
    gdb_path = Path(gdb_path).expanduser().resolve()

    if not gdb_path.exists():
        raise FileNotFoundError(f"GDB not found: {gdb_path}")
    
    log.info(f"📂 Loading GDB from: {gdb_path}")

    try:
        # If no layer is provided, show available layers and use the first one
        if layer is None:
            layers = fiona.listlayers(gdb_path)
            log.info(f"Available layers in {gdb_path.name}: {layers}")
            if not layers:
                raise ValueError(f"No layers found in {gdb_path}")
            layer = layers[0]
            log.info(f"Using first layer: '{layer}'")
        
        gdf = gpd.read_file(gdb_path, layer=layer)
        log.info(f" Loaded {len(gdf)} features from layer '{layer}'.")

        # Optional subsampling for quick EDA
        if subset_size and len(gdf) > subset_size:
            gdf = gdf.sample(n=subset_size, random_state=random_state) # TODO maybe change this random state/ get rid of sampling here?
            log.info(f"Subsampled to {len(gdf)} features for local testing.")

        return gdf
    except Exception as e:
        raise RuntimeError(f"Error reading GDB '{gdb_path}': {e}")
    
def summarize_numerical_features(
        gdf: gpd.GeoDataFrame,
        features: List[str]
) -> pd.DataFrame:
    """
    Compute descriptive statistics for selected numerical features.
    
    Args:
        gdf (gpd.GeoDataFrame): Input MapPLUTO GeoDataFrame.
        features (List[str]): List of numerical feature names to summarize.
    
    Returns:
        pd.DataFrame: Descriptive statistics for existing numerical features.
    """
    existing_features = [f for f in features if f in gdf.columns]
    missing_features = [f for f in features if f not in gdf.columns]

    if missing_features:
        log.warning(f"⚠️ Missing numerical columns: {missing_features}")

    if not existing_features:
        log.error("❌ No valid numerical features found in the GeoDataFrame.")
        return pd.DataFrame()

    stats = gdf[existing_features].describe()
    log.info(f"🧮 Computed summary statistics for {len(existing_features)} numerical features.")
    return stats

def summarize_categorical_features(
    gdf: gpd.GeoDataFrame,
    features: List[str],
    top_n: int = 10
) -> Dict[str, pd.Series]:
    """
    Compute top N value counts for categorical features.
    
    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        features (List[str]): List of categorical feature names to summarize.
        top_n (int): Number of top categories to return per feature.
    
    Returns:
        Dict[str, pd.Series]: Mapping of column name to its top N value counts.
    """
    existing_features = [f for f in features if f in gdf.columns]
    missing_features = [f for f in features if f not in gdf.columns]

    if missing_features:
        log.warning(f"⚠️ Missing categorical columns: {missing_features}")

    results = {}
    for f in existing_features:
        # dropna=False ensures we also see how many missing values exist,
        # which is useful for quality assessment in NYC datasets.
        counts = gdf[f].value_counts(dropna=False).head(top_n)
        results[f] = counts
        log.info(f"📊 Computed top {top_n} categories for '{f}'.")

    if not results:
        log.error("❌ No valid categorical features found in the GeoDataFrame.")
    return results


